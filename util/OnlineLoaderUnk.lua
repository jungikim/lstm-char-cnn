
-- Modified from https://github.com/karpathy/char-rnn
-- This version is for cases where one has already segmented train/val/test splits

local OnlineLoaderUnk = {}
local stringx = require('pl.stringx')
OnlineLoaderUnk.__index = OnlineLoaderUnk
utf8 = require 'lua-utf8'

function OnlineLoaderUnk.create(data_dir, batch_size, seq_length, max_word_l)
  local self = {}
  setmetatable(self, OnlineLoaderUnk)

  self.batch_size = batch_size
  self.seq_length = seq_length

  local train_file = path.join(data_dir, 'train.txt')
  local valid_file = path.join(data_dir, 'valid.txt')
  local test_file = path.join(data_dir, 'test.txt')
  self.input_files = {train_file, valid_file, test_file}
  local vocabfile = path.join(data_dir, 'vocab.txt')
  local stat_file = path.join(data_dir, 'stat.t7')

  -- construct a tensor with all the data
  if not path.exists(stat_file) then
    print('one-time setup: preprocessing input train/valid/test files in dir: ' .. data_dir)
    OnlineLoaderUnk.preprocess(vocabfile, self.input_files, batch_size, max_word_l, stat_file)
  end

  print('loading data files...')
  local stat_mapping = torch.load(stat_file)
  self.split_sizes = {}
  self.idx2word, self.word2idx, self.idx2char, self.char2idx, self.max_word_l,
     self.split_sizes[1], self.split_sizes[2], self.split_sizes[3] = table.unpack(stat_mapping)
  self.vocab_size = #self.idx2word
  print(string.format('Word vocab size: %d, Char vocab size: %d', #self.idx2word, #self.idx2char))
  
  self.dataFIterator={}
  self.dataFIterator[1] = io.open(self.input_files[1])
  self.dataFIterator[2] = io.open(self.input_files[2])
  self.dataFIterator[3] = io.open(self.input_files[3])

  self.batch_idx = {0,0,0}
  print(string.format('data load done. Number of batches in train: %d, val: %d, test: %d', self.split_sizes[1], self.split_sizes[2], self.split_sizes[3]))
  collectgarbage()
  return self
end

function OnlineLoaderUnk:reset_batch_pointer(split_idx, batch_idx)
  batch_idx = batch_idx or 0
  self.batch_idx[split_idx] = batch_idx
end

function OnlineLoaderUnk:next_batch(split_idx)
  -- split_idx is integer: 1 = train, 2 = val, 3 = test

  local batchSize = self.batch_size * self.seq_length

  local output_words = {} -- output tensors for train/val/test
  while #output_words < batchSize do
    local line = self.dataFIterator[split_idx]:read("*l")
    if line == nil then
      self.dataFIterator[split_idx]:close()
      self.dataFIterator[split_idx]=io.open(self.input_files[split_idx])
      self.batch_idx[split_idx] = 1 -- cycle around to beginning
      line = self.dataFIterator[split_idx]:read("*l")
    else
     self.batch_idx[split_idx] = self.batch_idx[split_idx] + 1
    end

--    line = stringx.replace(line, '<unk>', opt.tokens.UNK)
    function append(word)
      if self.word2idx[word]==nil then
        word = opt.tokens.UNK
      end
      table.insert(output_words, word)
    end
    append(opt.tokens.BOS)
    for rword in line:gmatch'([^%s]+)' do
      append(rword)
    end
    append(opt.tokens.EOS)
  end
  
  local output_word_tensor = torch.LongTensor(batchSize)
  local output_char_tensor = torch.ones(batchSize, self.max_word_l):long()

  for wIdx=1,batchSize do
    word = output_words[wIdx]
    output_word_tensor[wIdx] = self.word2idx[word]
    local l = utf8.len(word)
    local cIdx = 1
    for _, char in utf8.next, word do
      if cIdx == self.max_word_l then
        -- end-of-word symbol
        output_char_tensor[wIdx][cIdx] = self.char2idx[opt.tokens.END]
        break
      end
      
      char = utf8.char(char) -- save as actual characters
      if self.char2idx[char]==nil then
        output_char_tensor[wIdx][cIdx] = self.char2idx[opt.tokens.UNK] 
      else
        output_char_tensor[wIdx][cIdx] = self.char2idx[char]
      end
      cIdx = cIdx + 1
    end
  end
  
  local data = output_word_tensor
  local data_char = output_char_tensor

--  local len = data:size(1)
--  if len % (self.batch_size * self.seq_length) ~= 0 and split_idx < 3 then
--    data = data:sub(1, self.batch_size * self.seq_length * math.floor(len / (self.batch_size * self.seq_length)))
--  end
  local ydata = data:clone()
  ydata:sub(1,-2):copy(data:sub(2,-1))
  ydata[-1] = data[1]
  if split_idx < 3 then
    x_batch = data:view(self.batch_size, -1):split(self.seq_length, 2)
    y_batch = ydata:view(self.batch_size, -1):split(self.seq_length, 2)
    x_char_batch = data_char:view(self.batch_size, -1, self.max_word_l):split(self.seq_length,2)
    assert(#x_batch == #y_batch)
    assert(#x_batch == #x_char_batch)
  else --for test we repeat dimensions to batch size (easier but inefficient evaluation)
    x_batch = data:resize(1, data:size(1)):expand(self.batch_size, data:size(2))
    y_batch = ydata:resize(1, ydata:size(1)):expand(self.batch_size, ydata:size(2))
    data_char = data_char:resize(1, data_char:size(1), data_char:size(2))
    x_char_batch = data_char:expand(self.batch_size, data_char:size(2), data_char:size(3))
  end

  return x_batch[1], y_batch[1], x_char_batch[1]
end

function OnlineLoaderUnk.preprocess(vocabfile, input_files, batch_size, max_word_l, stat_file)
  print('Processing text...')
  local tokens = opt.tokens -- inherit global constants for tokens
  local f
  local max_word_l_tmp = 0 -- max word length of the corpus
  local idx2word = {opt.tokens.ZEROPAD, opt.tokens.UNK} -- unknown word token
  local word2idx = {}; word2idx[opt.tokens.ZEROPAD] = 1; word2idx[opt.tokens.UNK] = 2
  local idx2char = {opt.tokens.ZEROPAD, opt.tokens.START, opt.tokens.END, opt.tokens.UNK} -- zero-pad, start-of-word, end-of-word tokens
  local char2idx = {}; char2idx[opt.tokens.ZEROPAD] = 1; char2idx[opt.tokens.START] = 2; char2idx[opt.tokens.END] = 3; char2idx[opt.tokens.UNK] = 4;
  local split_sizes = {}

-- load vocab file
  f = io.open(vocabfile, 'r')
  for line in f:lines() do
    if word2idx[line] == nil then
      idx2word[#idx2word + 1] = line -- create word-idx/idx-word mappings
      word2idx[line] = #idx2word
    end
  end
  f:close()

  -- first go through train/valid/test to get max word length
  -- if actual max word length is smaller than specified
  -- we use that instead. this is inefficient, but only a one-off thing so should be fine
  -- also counts the number of tokens
  for	split = 1,3 do -- split = 1 (train), 2 (val), or 3 (test)
    f = io.open(input_files[split], 'r')
    local numLines = 0
    local counts = 0
    for line in f:lines() do
      numLines = numLines + 1
--      line = stringx.replace(line, '<unk>', opt.tokens.UNK) -- replace unk with a single character
      counts = counts + 1 -- BOS
      for word in line:gmatch'([^%s]+)' do
        max_word_l_tmp = math.max(max_word_l_tmp, utf8.len(word) + 2) -- add 2 for start/end chars
        counts = counts + 1

        local l = utf8.len(word)
        for _, char in utf8.next, word do
          char = utf8.char(char) -- save as actual characters
          if char2idx[char]==nil then
            idx2char[#idx2char + 1] = char -- create char-idx/idx-char mappings
            char2idx[char] = #idx2char
          end
        end

      end
      counts = counts + 1 -- EOS
    end
    f:close()
    split_sizes[split] = math.ceil(numLines / batch_size) 
  end

  print('Max word length is: ' .. max_word_l_tmp)
  print(string.format('batch count: train %d, val %d, test %d', split_sizes[1], split_sizes[2], split_sizes[3]))

  -- if actual max word length is less than the limit, use that
  max_word_l = math.min(max_word_l_tmp, max_word_l)

  print "done"
  -- save output preprocessed files
  print('saving ' .. stat_file)
  torch.save(stat_file, {idx2word, word2idx, idx2char, char2idx, max_word_l, split_sizes[1], split_sizes[2], split_sizes[3]})
end

return OnlineLoaderUnk

