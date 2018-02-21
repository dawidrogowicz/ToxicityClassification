import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle
import os
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from model import model
from collections import Counter

wnl = WordNetLemmatizer()

STOP_WORDS = set(stopwords.words('english'))
CATEGORIES = ('toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate')
TRAIN_PATH = 'dataset/train.csv'
TEST_PATH = 'dataset/test.csv'
DICTIONARIES_PATH = 'pickles/dictionaries.pickle'
DATA_X_PATH = 'pickles/data_x.pickle'


def token_valid(token):
    return ((token not in STOP_WORDS)
            and (len(token) > 2)
            and (len(token) < 20)
            and re.match(r'^[a-z]+$', token))


def create_lexicon(sentence_list, n_tokens=10000):
    _lexicon = list()
    for sentence in tqdm(sentence_list):
        words = word_tokenize(sentence.lower())
        words = [wnl.lemmatize(token) for token in words if token_valid(token)]
        _lexicon.extend(words)

    # use only n most common words
    _lexicon = set([count[0] for count in Counter(_lexicon).most_common(n_tokens)])
    return _lexicon


def to_one_hot(sentence, _lexicon):
    tokens = word_tokenize(sentence.lower())
    tokens = set([wnl.lemmatize(token) for token in tokens if re.match(r'^[a-z]+$', token)])
    output = []
    for lexicon_token in _lexicon:
        if lexicon_token in tokens:
            output.append(1)
        else:
            output.append(0)
    return output


def create_dictionary(sentences):
    _lexicon = create_lexicon(sentences)
    _dictionary = dict()
    for entry in _lexicon:
        _dictionary[entry] = len(_dictionary)
    _reverse_dictionary = dict(zip(_dictionary.values(), _dictionary.keys()))
    return _dictionary, _reverse_dictionary


df = pd.read_csv(TRAIN_PATH)
# limit number of samples until model is ready
df = df[:10000]

data_x = df['comment_text']
data_y = np.array(df.iloc[:, 2:])

# If dictionaries file exists load it,
# if not create new dictionaries and save them to the file
if os.path.exists(DICTIONARIES_PATH):
    with open(DICTIONARIES_PATH, 'rb') as f:
        dictionary, reverse_dictionary = pickle.load(f)
    print('dictionaries loaded')
else:
    dictionaries = create_dictionary(data_x)
    dictionary, reverse_dictionary = dictionaries
    with open(DICTIONARIES_PATH, 'wb') as f:
        pickle.dump(dictionaries, f)
    print('dictionaries created')

print(dictionary)
print(reverse_dictionary)

# # If features file exists load it,
# # if not process new features and save it to the file
# if os.path.exists(DATA_X_PATH):
#     with open(DATA_X_PATH, 'rb') as f:
#         data_x = pickle.load(f)
#     print('features loaded')
# else:
#     data_x = np.array(embedd_words(data_x, lexicon))
#     with open(DATA_X_PATH, 'wb') as f:
#         pickle.dump(data_x, f)
#     print('features processed')
#
# train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=.3, random_state=1)
# length = len(lexicon)
#
# model = model((None, length))
# model.fit(train_x, train_y, 10)
#
# acc = model.evaluate(test_x, test_y)
# print(acc)
