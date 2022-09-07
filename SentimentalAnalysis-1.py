import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

tweet_df = pd.read_csv('train-v2.tsv', names=['sentiment', 'text'], sep='\t')
tweet_df

# Training/Testing dataset
sentiment = tweet_df['sentiment'].values
text = tweet_df['text'].values
sentiment_train, sentiment_test, text_train, text_test = train_test_split(sentiment, text, test_size=0.1, random_state=42)

# Generate vocabulary by vactorizing from the training data
vectorizer = CountVectorizer(min_df=0, lowercase=False,stop_words='english') # will also remove punctuation or stop words
vectorizer.fit(text_train) ### fit on the whole dataset

# createing feature vector for each sentence
X_train = vectorizer.transform(text_train).toarray()
X_test = vectorizer.transform(text_test).toarray()

# Build baseline model using NB algo's
model1 = MultinomialNB()
model1.fit(X_train, sentiment_train)
score = model1.score(X_test, sentiment_test)
print("Accuracy of MultinomialNB model:", score)




