# Sentiment-Analysis
 Sentiment-Analysis with various techniques
 
 Follow Assignment-1:-
 Two methods were adopted for building the classifer(MultibomialNB & Word Embedding).
 - In the first,words are vectorized and Multinomial Naive Bayesian Classifier was used.
 - In the second approach these words were tokenized and fed to a word embedding deep learning model.

Introduction: This challenge involves sentiment analysis of tweets written in the Welsh language (Links to an external site.). In the absence of a gold-standard training corpus for Welsh, I've used the presence or absence of “happy” and “sad” emoji/emoticons as a proxy for sentiment. For example:

    “Happy”: 😃, 😄, 😅, 😍, :)
    “Sad”: 😞, 😟, 😡, :(

We then define a tweet to have positive sentiment if it contains at least one happy emoji/emoticon, and no sad emoji/emoticon, and we define negative sentiment analogously.

Training/Test Set: I have a large collection of about 5.5 million Welsh language tweets gathered as part of the Indigenous Tweets project (Links to an external site.). It turns out that positive sentiment tweets are about 5 times more common than negative sentiment tweets, and so to make the machine learning challenge more interesting, I decided to balance out the training data to a 50/50 split. I therefore chose a random sample of 45000 positive sentiment tweets, and a random sample of 45000 negative sentiment tweets. These were combined into a single list from which I chose a random sample of 80000 as training data, setting aside the remaining 10000 as test data.

The training data is provided in a TSV file, with one training example per line. The first field contains the label: 0 for negative sentiment and 1 for positive sentiment. The second field contains the text of the tweet, preprocessed as described below. The test data is in the same format, but you won't have access to that file! If you need development data in order to tune the parameters of your model(s), you'll need to take a subset of the training data for that.

*** Download Training Data (3MB) Download Download Training Data (3MB)

Preprocessing: To preserve some anonymity, all usernames were converted to the token “@USER”. All URLs were converted to the token “{URL}”. All whitespace characters (including newline/tab) were converted to ASCII space (U+0020), and then multiple spaces were collapsed to a single space.

Evaluation: The goal is to train a system that achieves the highest possible accuracy (percentage of correctly-predicted labels) on the test set.

Deliverables and Grading: You will implement two solutions to the given challenge; one “from scratch” implementation that doesn't use any NLP or machine learning libraries, and one “anything goes” implementation in which you're free to use additional libraries, code, data, or linguistic knowledge. Details regarding the format of your submission and the grading scheme are provided in the Rubric for NLP Challenges document.

For this particular challenge, the code you submit should read the training data file train.tsv from the same directory it's launched from. There is a sample notebook (Links to an external site.) in the course github repo that you can use to begin your work (of course you won't be able to do the final evaluation with the test.tsv file, but you can (and should!) track your progress by setting aside a development set). You are free to restructure the code however you'd like, but I'd like the last line to call the “evaluate” function which displays the accuracy of your two algorithms on the test set.

How to submit: The best way to submit your code is by committing it to a git repository (on github or on the CS Department gitlab instanceLinks to an external site. for example) that I'd be able to clone.
