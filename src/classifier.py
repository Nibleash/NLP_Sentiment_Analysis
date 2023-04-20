import re
import torch
import pandas as pd
from typing import List

import nltk
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_sentences(df):
    clean_df = df.copy()
    print(f'\n{clean_df.head()}\n')

    # Sentence and target to lower to avoid capital letters issue.
    clean_df['target'] = clean_df['target'].apply(lambda x: x.lower())
    clean_df['sentence'] = clean_df['sentence'].apply(lambda x: x.lower())

    # Remove punctuation using regex.
    clean_df['target'] = clean_df['target'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
    clean_df['sentence'] = clean_df['sentence'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

    # Remove numbers using regex.
    clean_df['target'] = clean_df['target'].apply(lambda x: re.sub(r'\d+', '', x))
    clean_df['sentence'] = clean_df['sentence'].apply(lambda x: re.sub(r'\d+', '', x))

    # Lemmatize the verbs.
    clean_df['target'] = clean_df['target'].apply(lambda x: " ".join([WordNetLemmatizer().lemmatize(word, 'v') for word in x.split()]))
    clean_df['sentence'] = clean_df['sentence'].apply(lambda x: " ".join([WordNetLemmatizer().lemmatize(word, 'v') for word in x.split()]))

    return clean_df


class Classifier:
    """The Classifier"""

    def train(self, train_filename: str, dev_filename: str =None, device: torch.device =device):
        """Trains the classifier model on the training set stored in file train_filename"""

        # We load the data and lower the text
        data = pd.read_csv(train_filename, sep='\t', header=None, names=['polarity', 'aspect', 'target', 'position', 'sentence'])
        clean_data = clean_sentences(data)
        
        # We create a BOW vector
        self.restaurant_vect = CountVectorizer(min_df=1, tokenizer=nltk.word_tokenize)
        reviews_counts = self.restaurant_vect.fit_transform(clean_data.sentence)
    
        # We transform the BOW vector with the tfidf scores
        self.tfidf_transformer = TfidfTransformer()
        reviews_tfidf = self.tfidf_transformer.fit_transform(reviews_counts)

        # Train a Linear Support Vector Classifier
        self.clf = LinearSVC().fit(reviews_tfidf, clean_data.polarity)


    def predict(self, data_filename: str, device: torch.device=device) -> List[str]:
        """Predicts class labels for the input instances in file 'data_filename'
        Returns the list of predicted labels
        """
 
        # We load the test data and lower the text
        data_test = pd.read_csv(data_filename, sep = "\t", names = ['polarity', 'aspect', 'target', 'position', 'sentence'])
        clean_test_data = clean_sentences(data_test)
        
        # We create a BOW vector
        reviews_new_counts = self.restaurant_vect.transform(clean_test_data.sentence)
        
        # We transform the BOW vector with the tfidf scores
        reviews_new_tfidf = self.tfidf_transformer.transform(reviews_new_counts)
        
        # We make a prediction with the classifier
        self.pred = self.clf.predict(reviews_new_tfidf)
        
        return self.pred
