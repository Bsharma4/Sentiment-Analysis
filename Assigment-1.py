import random
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# loading data in data frame
tweet_df = pd.read_csv('train-v2.tsv', names=['sentiment', 'text'], sep='\t')
sentiment = tweet_df['sentiment'].values
text = tweet_df['text'].values
# Spiliting data
tweet_df = pd.read_csv('train-v2.tsv', names=['sentiment', 'text'], sep='\t')
sentiment = tweet_df['sentiment'].values
text = tweet_df['text'].values
# Preparing Training/Testing dataset
sentiment_train, sentiment_test, text_train, text_test = train_test_split(sentiment, text, test_size=0.5, random_state=42)
 
def predict_from_scratch(tweet):
  vectorizer = CountVectorizer(min_df=0, lowercase=False,stop_words='english') # will also remove punctuation or stop words
  vectorizer.fit(text_train) ### fit on the whole dataset
  # createing feature vector for each sentence
  X_train = vectorizer.transform(text_train).toarray()
  X_test = vectorizer.transform(text_test).toarray()
  X_pred = vectorizer.transform(tweet).toarray()
  # Build baseline model using NB algo's
  model1 = MultinomialNB()
  model1.fit(X_train, sentiment_train)
  score = model1.score(X_test, sentiment_test)
  print("Accuracy of MultinomialNB model:", score)
  prediction = model1.predict(X_pred)
  if (prediction>=5): return 1
  else: return 0

def predict_anything_goes(tweet):
  from keras_preprocessing.sequence import pad_sequences
  from keras.preprocessing.text import Tokenizer
  from keras.models import Sequential
  from keras.layers import Embedding
  from keras import layers
  # Tokenize the text for processing
  tokenizer = Tokenizer(num_words=70000)
  tokenizer.fit_on_texts(text_train)
  X_train = tokenizer.texts_to_sequences(text_train)
  X_test = tokenizer.texts_to_sequences(text_test)
  X_predict = tokenizer.texts_to_sequences(tweet)
  maxlen = 100
  X_train_pad = pad_sequences(X_train, padding='post', maxlen=maxlen)
  X_test_pad = pad_sequences(X_test, padding='post', maxlen=maxlen)
  X_predict_pad = pad_sequences(X_predict, padding='post', maxlen=maxlen)
  # building word embedding layer
  embedding_dim = 50 ## embedding size
  vocab_size = len(tokenizer.word_index) + 1 # addition value 0 for padding
  model = Sequential()
  model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen, embeddings_initializer=None)) ## Set embeddings_initializer to some other pre-trained weights for transfer learning
  model.add(layers.GlobalMaxPool1D())
  model.add(layers.Dense(10, activation='relu'))
  model.add(layers.Dense(1, activation='sigmoid')) 
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  model.summary()
  ## Start training the input data, the input is indices of words
  history = model.fit(X_train_pad, sentiment_train, epochs=5, verbose=True, validation_split=0.1, batch_size=10)
  ## calculating loss and accuracy on trained model
  loss, accuracy = model.evaluate(X_train_pad, sentiment_train, verbose=False)
  print("Training Accuracy: {:.4f}".format(accuracy))
  loss, accuracy = model.evaluate(X_test_pad, sentiment_test, verbose=False)
  print("Testing Accuracy: {:.4f}".format(accuracy))
  ## Prediction
  p = model.predict(X_predict_pad)
  if (p>=5): return 1
  else: return 0
    

def evaluate():
  t = tweet_df.sample()
  p_S = predict_from_scratch(t['text'])
  p_A = predict_anything_goes(t['text'])
  print("tweet: ", t['text'])

  if (t['sentiment'].values>=5): print("actual sentiment: Positive")
  else: print("actual sentiment: Negative")
  
  if (p_S==1): print("predicted sentiment from MultinomialNB model: Positive")
  else: print("predicted sentiment from MultinomialNB model: Negative ")

  if (p_A==1): print("predicted sentiment from word embedded model: Positive")
  else: print("predicted sentiment from word embedded model: Negative ")

  return 0

evaluate()