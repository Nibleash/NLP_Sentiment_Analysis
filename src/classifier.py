import re
import torch
import pandas as pd
import numpy as np
from typing import List
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import nltk
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer


def clean_sentences(df):
    clean_df = df.copy()

    for column in ['target', 'sentence']:
        # Sentence and target to lower to avoid capital letters issue.
        clean_df[column] = clean_df[column].apply(lambda x: x.lower())
        # Remove punctuation using regex.
        clean_df[column] = clean_df[column].apply(lambda x: re.sub(r'[^\w\s]', '', x))
        # Remove numbers using regex.
        clean_df[column] = clean_df[column].apply(lambda x: re.sub(r'\d+', '', x))
        # Lemmatize the verbs.
        clean_df[column] = clean_df[column].apply(lambda x: " ".join([WordNetLemmatizer().lemmatize(word, 'v') for word in x.split()]))

    return clean_df


class Classifier():

    def __init__(self):
        self.mapping_dict = {'positive': 2, 'neutral': 1, 'negative': 0}
        self.reverse_mapping_dict = {v: k for k, v in self.mapping_dict .items()}
        self.tokenizer_self_bert = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3,
            output_attentions=False, output_hidden_states=False)
        self.batch_size = 16
        self.epochs = 5
        self.token_length = 128
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = 5e-5, eps = 1e-08) # Very low learning rate to finetune the model don't disturb too much the pretrained weights.


    def tokenize(self, df):
        token_df = df.copy()
        token_df['bert_encoded_dict'] = token_df['bert_encoded'].apply(
            lambda x: self.tokenizer_self_bert.encode_plus(text=x, add_special_tokens=True,
            padding='max_length', max_length=self.token_length, return_attention_mask=True))
        token_df = pd.concat([token_df.drop(['bert_encoded_dict'], axis=1), token_df['bert_encoded_dict'].apply(pd.Series)], axis=1)
        del token_df['token_type_ids']
        return token_df


    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the classifier model on the training set stored in file train_filename.
        """

        # We load the data and clean the text
        data = pd.read_csv(train_filename, sep='\t', header=None, names=['polarity', 'aspect', 'target', 'position', 'sentence'])
        clean_data = clean_sentences(data)

        # Before encoding, we need to aggregate all the text we want to consider using BERT specific markers
        clean_data['bert_encoded'] = clean_data['sentence'].astype(str)  + '[SEP]' + clean_data['aspect'].astype(str) + '[SEP]' + clean_data['target'].astype(str)

        # Now we need to tokenize the text using BertTokenizer and to format the input vectors
        tokenize_data = self.tokenize(clean_data)
        tokenize_data['polarity'] = tokenize_data['polarity'].map(self.mapping_dict)
        token_ids = torch.tensor(np.vstack(tokenize_data['input_ids'].apply(np.ravel))).to(device)
        token_attention = torch.tensor(np.vstack(tokenize_data['attention_mask'].apply(np.ravel))).to(device)
        token_labels = torch.tensor(tokenize_data['polarity'].values).to(device)

        # Train set and prepare DataLoader
        train_set = TensorDataset(token_ids, token_attention, token_labels)
        train_dataloader = DataLoader(train_set, sampler=RandomSampler(train_set), batch_size=self.batch_size)

        # ---------- TRAINING LOOP ----------
        self.model = self.model.to(device)
        for epoch in range(self.epochs):
            self.model.train()
            tr_loss = 0

            for step, batch in enumerate(train_dataloader):
                b_input_ids, b_input_mask, b_labels = batch
                self.optimizer.zero_grad()
                # Forward pass
                train_output = self.model(b_input_ids, token_type_ids=None,
                                          attention_mask=b_input_mask, labels=b_labels)
                # Backward pass
                train_output.loss.backward()
                self.optimizer.step()
                # Update tracking variables
                tr_loss += train_output.loss.item()
            print(f'Epoch {epoch}: training loss = {tr_loss}')


    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """
        Predicts class labels for the input instances in file 'data_filename'.
        Returns the list of predicted labels.
        """
        
        # We load the test data and clean the text
        data_test = pd.read_csv(data_filename, sep = "\t", names = ['polarity', 'aspect', 'target', 'position', 'sentence'])
        clean_test_data = clean_sentences(data_test)

        # Again we use BertTokenizer to tokenize the text: target words and sentences
        clean_test_data['bert_encoded'] = clean_test_data['sentence'].astype(str)  + '[SEP]' + clean_test_data['aspect'].astype(str) + '[SEP]' + clean_test_data['target'].astype(str)
        tokenize_test_data = self.tokenize(clean_test_data)
        
        # Format the test input vectors and prepare DataLoader
        test_token_ids = torch.tensor(np.vstack(tokenize_test_data['input_ids'].apply(np.ravel))).to(device)
        test_token_attention = torch.tensor(np.vstack(tokenize_test_data['attention_mask'].apply(np.ravel))).to(device)
        test_set = TensorDataset(test_token_ids, test_token_attention)
        test_dataloader = DataLoader(test_set, sampler=SequentialSampler(test_set), batch_size=self.batch_size)

        # ---------- INFERENCE LOOP ----------
        self.model.eval()
        self.pred = []
        for batch in test_dataloader:
            b_input_ids, b_input_mask = batch
            with torch.no_grad():
                # Forward pass
                eval_output = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            pred_polarity = np.argmax(eval_output.logits.cpu().detach().numpy(), axis=1)
            self.pred += [self.reverse_mapping_dict[x] for x in pred_polarity]

        return self.pred