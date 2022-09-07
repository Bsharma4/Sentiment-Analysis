import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.preprocessing.sequence import pad_sequences

tweet_df = pd.read_csv('train-v2.tsv', names=['sentiment', 'text'], sep='\t')
tweet_df

# Training/Testing dataset
sentiment = tweet_df['sentiment'].values
text = tweet_df['text'].values
sentiment_train, sentiment_test, text_train, text_test = train_test_split(sentiment, text, test_size=0.3, random_state=42)

# Step 6.1: Practice the tokenize on the first two sequences
## Step 6.3: Text tokenize to get vocabulary
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(text_train)
X_train = tokenizer.texts_to_sequences(text_train)
X_test = tokenizer.texts_to_sequences(text_test)

from keras.preprocessing.sequence import pad_sequences
maxlen = 100
X_train_pad = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test_pad = pad_sequences(X_test, padding='post', maxlen=maxlen)
X_train_pad

## Step 6.3: Text encoding using one-hot encoding
from sklearn.preprocessing import OneHotEncoder
import numpy as np
enc = OneHotEncoder(sparse=False)
all_labels = X_train_pad.reshape(-1,)
all_labels
enc.fit(all_labels.reshape(len(all_labels),1)) 

X_train_pad_onehot = []

for sentence in X_train_pad:
    X_train_pad_onehot.append(enc.transform(sentence.reshape(maxlen,1)))

X_test_pad_onehot = []
for sentence in X_test_pad:
   X_test_pad_onehot.append(enc.transform(sentence.reshape(maxlen,1)))

X_train_pad_onehot = np.asarray(X_train_pad_onehot)
X_test_pad_onehot = np.asarray(X_test_pad_onehot)

# Task: Write codes to check the shape of two matrix
print("X_train_pad_onehot shape: ", X_train_pad_onehot.shape)
print("X_test_pad_onehot shape: ", X_test_pad_onehot.shape)

# step 6.4.1: To feed into dense layer, we need flatten matrix to get tabular data format
X_data_flatten = X_train_pad_onehot.reshape(X_train_pad_onehot.shape[0],100*1757)
X_test_flatten = X_test_pad_onehot.reshape(X_test_pad_onehot.shape[0],100*1757)
print("Training matrix shape", X_data_flatten.shape)
print("Testing matrix shape", X_test_flatten.shape)

# step 6.4.2:build basic neural network module
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dropout,Activation
from keras.layers import BatchNormalization,Dense
from keras.regularizers import l2
from keras import initializers,Sequential
import numpy as np

keras_callbacks = [
 EarlyStopping(monitor='val_loss', patience=10, mode='min', min_delta=0.0001),
 ModelCheckpoint('./checkmodel.h5', monitor='val_loss', save_best_only=True, mode='min')
]

def build_model(n_layers = 2, n_neurons = 1000,initializer='uniform'):
    model = Sequential() # create Sequential model
    for i in range(n_layers-1):
        model.add(Dense(n_neurons, kernel_initializer=initializer))
        model.add(BatchNormalization()) ## add batch normalization before activation
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
    model.add(Dense(1, activation = 'sigmoid', kernel_initializer=initializer)) 
    return model

## Step 6.4.3 Training the network
model = build_model(n_layers = 5, n_neurons = 1000,initializer='uniform')
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ['accuracy'])
train_history = model.fit(X_data_flatten,sentiment_train, validation_split=0.1, batch_size = 10, epochs = 50, callbacks=keras_callbacks)

## Step 6.4.4: Evaluate the prediction
print("Accuracy of neural network model:", model.evaluate(X_test_flatten, sentiment_test)[1])




