import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder


tweet_df = pd.read_csv('train-v2.tsv', names=['sentiment', 'text'], sep='\t')
tweet_df

# Training/Testing dataset
sentiment = tweet_df['sentiment'].values
text = tweet_df['text'].values
sentiment_train, sentiment_test, text_train, text_test = train_test_split(sentiment, text, test_size=0.25, random_state=42)

# Step 6.1: Practice the tokenize on the first two sequences
## Step 6.3: Text tokenize to get vocabulary
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(text_train)
X_train = tokenizer.texts_to_sequences(text_train)
X_test = tokenizer.texts_to_sequences(text_test)

maxlen = 100
X_train_pad = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test_pad = pad_sequences(X_test, padding='post', maxlen=maxlen)
X_train_pad

## Step 6.3: Text encoding using one-hot encoding
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

