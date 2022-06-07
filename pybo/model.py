import tensorflow as tf
import numpy as np
import re
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.sequence import pad_sequences

okt = Okt()
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
tokenizer = tf.keras.preprocessing.text.Tokenizer()
loaded_model = tf.keras.models.load_model('../models/classification_fuck_model.h5')
max_len = 30

def sentiment_predict(new_sentence):
  new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', new_sentence)
  new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
  new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
  encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
  pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
  score = float(loaded_model.predict(pad_new)) # 예측
  if(score > 0.5):
    # return "{:.2f}% 확률로 긍정 리뷰입니다.\n".format(score * 100)
    print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(score * 100))
  else:
    # return "{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - score) * 100)
    print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - score) * 100))

def toxicCommentClassficiationPredict(inputText):
    outputText = sentiment_predict(inputText)
    return outputText