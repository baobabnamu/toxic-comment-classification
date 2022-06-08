import pickle
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.optim import Adam, SGD
import tensorflow as tf
import numpy as np
import re
from konlpy.tag import Okt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import itertools

# CLASSIFICATION
okt = Okt()
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
tokenizer = Tokenizer()
loaded_model = tf.keras.models.load_model('../models/classification_model.h5')
max_len = 30

def sentiment_predict(new_sentence):
  new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', new_sentence)
  new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
  new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
  encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
  pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
  score = float(loaded_model.predict(pad_new)) # 예측
  if(score > 0.5):
    return "{:.2f}% 확률로 긍정 리뷰입니다.\n".format(score * 100)
  else:
    return "{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - score) * 100)


# FILTERING
with open('../models/params.pkl', 'rb') as pkl:
    params = pickle.load(pkl)
word2idx = params['word2idx']
idx2word = params['idx2word']
SENTENCE_LENGTH = params['sentence_length']
torch.manual_seed(42)

class Discriminator(nn.Module):
    def __init__(self, vocab_size, embed_size, c_size, kernel_num, kernel_sizes):
        super().__init__()
        
        self.embedding = nn.Embedding(
            vocab_size, embed_size, padding_idx=0
        )
        self.conv_list = [
            nn.Conv1d(embed_size, kernel_num, kernel_size=kernel_size) 
            for kernel_size in kernel_sizes
        ]
        self.convs = nn.ModuleList(self.conv_list)
        
        self.maxpools = nn.ModuleList([
            nn.MaxPool1d(kernel_size) 
            for kernel_size in kernel_sizes
        ])
        
        self.linear = nn.Linear(2200, c_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        batch_size = x.size(0)
        embedded = self.embedding(x)
        embedded = embedded.transpose(1,2)
        
        pools = []
        for conv, maxpool in zip(self.convs, self.maxpools):
            feature_map = conv(embedded)
            pooled = maxpool(feature_map)
            pools.append(pooled)
            
        conv_concat = torch.cat(pools, dim=-1).view(batch_size, -1)
        conv_concat = self.dropout(conv_concat)
        logits = self.linear(conv_concat)
        return self.softmax(logits)

D = Discriminator(
    vocab_size=len(word2idx), 
    embed_size=128, 
    c_size=2, 
    kernel_num=100, 
    kernel_sizes=[2,3,4,5]
)
D.load_state_dict(torch.load('../models/D_180115.pth', map_location='cpu'))
D.eval()        

def process_sentences(sentences, word2idx, sentence_length=20, padding='<PAD>'):
    sentences_processed = []
    for sentence in sentences:
        if len(sentence) > sentence_length:
            fixed_sentence = sentence[:sentence_length]
        else:
            fixed_sentence = sentence + [padding]*(sentence_length - len(sentence))
        
        sentence_idx = [word2idx[word] if word in word2idx.keys() else word2idx['<UNK>'] for word in fixed_sentence]
        
        sentences_processed.append(sentence_idx)

    return sentences_processed

def clean(s):
    ss = re.sub('[^가-힣ㄱ-ㅎㅏ-ㅣA-Za-z0-9]+', '', s)
    ssss = ''.join(ch if len(list(grouper)) == 1 else ch*2 for ch, grouper in itertools.groupby(ss))
    return ssss

def do_inference(raw_sentences, print_clean=False):
  clean_sentences = [clean(s) for s in raw_sentences]
  sentences = [list(''.join(clean_sentence.split())) for clean_sentence in clean_sentences]
  infer_sentences_processed = process_sentences(sentences, word2idx, sentence_length=SENTENCE_LENGTH)
  data = torch.LongTensor(infer_sentences_processed)
  log_probs = D(Variable(data))
  probs = log_probs.exp()
  return probs

from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer()

def spacing_example(example):
    length = len(example.split())
    if length < 2:
        spaced = ' '.join([c for c in example.replace(' ', '')])
    else:
        spaced = example
    return spaced

def limer(example):
  try:
    buff = example
    exp = explainer.explain_instance(spacing_example(example), lambda s: do_inference(s, True).detach().numpy(), top_labels=1)
    exp.show_in_notebook()
    test = exp.as_list()
    for i in test:
      if i[1] >= 0.09:
        buff= buff.replace(str(i[0]),"*"*len(i[0]))
    return buff
    
  except:
    return example

def toxicCommentClassficiationPredict(inputText):
    outputText = sentiment_predict(inputText)
    return outputText

def toxicCommentFilteringPredict(inputText):
    outputText = limer(inputText)
    return outputText