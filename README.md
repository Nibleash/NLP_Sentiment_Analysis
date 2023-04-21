# NLP Assignment: Aspect-Term Polarity Classification in Sentiment Analysis 🌊🔥

Made by Martin PONCHON, Ethan SETROUK, Robinson DENEVE and Ugo DEMY.

## Dataset augmentation

Our datasets are TSV files with one instance per line, each line containing the polarity of the opinion, the aspect category on which the opinion is expressed, a specific target term, the character offsets of the term (start:end), and the sentence in which the term occurs and the opinion is expressed.

Here is an example of a line:

negative SERVICE#GENERAL Wait staff 0:10 Wait staff is blantently unappreciative of your business but its the best pie on the UWS!

This means that the opinion polarity regarding the target term *"wait staff"*, which has
the aspect category **SERVICE#GENERAL**, is **negative**.

Since we observed that our training dataset was highly unblanced with the 'positive' class representing a large proportion of our dataset, we considered data augmentation to increase the number of data points in the 'neutral' and 'negative' classes. To do so, we used back translation, which consists in translating content from english to french and then back to english multiple times, thus obtaining a slightly different sentence. We also added new sentences, annotated manually.

| Class  | Before data augmentation | After data augmentation |
| ------------- | ------------- | ------------- |
| Positive  | 70.2% |  64.6%  |
| Neutral  | 3.8%  | 7.3% |
| Negative | 26.0%  | 28.1% |

## Data preprocessing and cleaning

Regarding the data preprocessing and cleaning, we did not do too much since we use a contextual model (BERT) for the classification and operations like stopwords removal does not always help in these cases. Thus, we mainly focused on the sentences and the target words, performing 3 steps:
- Put everything to lower-case letters.
- Using regex, filter the punctuation and the digits.
- Lemmatize the verbs to extract only the core information of the word from it.

## Tokenization using BERT pre-trained model

For tokenisation and classification, we used the *bert-base-uncased*, which is a pretrained model on English language using a masked language modeling (MLM) objective. It has 110M parameters and a size of ~9GB (GPU RAM).
Also, regarding the tokenized vector size, we chose to keep the default token size for BERT: 128.

To specify the aspect-term to the model, we tokenize the following:

`sentence[SEP]aspect[SEP]target`

For example, to determine the opinion polarity regarding the target term "wait staff" in the example above, we would tokenize:

`wait staff is blantently unappreciative of your business but its the best pie on the uws[SEP]SERVICE#GENERAL[SEP]wait staff`

The encode_plus method allows us to tokenize the sentence, but it also
- Adds special tokens *[CLS]* and *[SEP]* at the start and at the end of the sentence
- Pads or truncates to the maximum length of our sentences
- Maps each token to its ID
- Creates and returns the attention mask

## Sentiment analysis with BERTForSequenceClassification

We trained the BertSequenceClassifier model on Colab GPU since we do not have one on our computers and using the CPU takes too much ressources and time.
The three polarity labels positive, negative or neutral are encoded as integers 2, 1 and 0 respectively. Indeed, we tried 1, 0 and -1 but, since we use the Cross Entropy Loss, the encoded labels' values have to be between 0 and the number of classes.

## Results

We trained the classifier using a Cross Entropy Loss.
Using the described classifier trained on 4 epochs, we obtained a 83.03% (+- 1.43) accuracy on the dev set.
