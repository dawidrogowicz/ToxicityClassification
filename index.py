import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle
import os
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from embeddings import train_embeddings, visualize_embeddings
from collections import Counter

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
wnl = WordNetLemmatizer()

# CONSTANTS
STOP_WORDS = set(stopwords.words('english'))
CATEGORIES = ('toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate')
TRAIN_PATH = 'dataset/train.csv'
TEST_PATH = 'dataset/test.csv'
DICTIONARIES_PATH = 'pickles/dictionaries.pickle'
DATA_X_PATH = 'pickles/data_x.pickle'
EMBEDDINGS_PATH = 'pickles/embeddings.pickle'
LEXICON_PATH = 'pickles/lexicon.pickle'
EMBED_DATA_X_PATH = 'pickles/embed_data_x.pickle'


# FUNCTIONS
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
    _lexicon = ['UNKNOWN'] + [count[0] for count in Counter(_lexicon).most_common(n_tokens - 1)]
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


def create_dictionary(_lexicon):
    _dictionary = dict()
    for entry in _lexicon:
        _dictionary[entry] = len(_dictionary)
    _reverse_dictionary = dict(zip(_dictionary.values(), _dictionary.keys()))
    return _dictionary, _reverse_dictionary


def convert_sentences(sentence_list, _dictionary):
    data = list()
    max_len = 0
    for sentence in tqdm(sentence_list):
        words = word_tokenize(sentence.lower())
        words = [_dictionary.get(wnl.lemmatize(token), 0) for token in words if token_valid(token)]

        if max_len < len(words):
            words = ';;;'.join(map(str, words))
            words = re.sub(r'(.+?)\1{3,}', r'\1' * 3, words).split(';;;')
            if (len(words) < 1) or (words == ['']):
                words = [0]
            words = list(map(int, words))
            if max_len < len(words):
                max_len = len(words)

        data.append(words)

    data = [np.pad(tokens,
                   (-(-(max_len - len(tokens)) // 2), (max_len - len(tokens)) // 2),
                   'constant',
                   constant_values=(0, 0)) for tokens in data]

    return data


# START
df = pd.read_csv(TRAIN_PATH)
# limit number of samples until model is ready
# df = df[:10000]

data_x = df['comment_text']
data_y = np.array(df.iloc[:, 2:])

del df

# LEXICON
# If lexicon file exists, load it
# else, create new lexicon and save it to the file
if os.path.exists(LEXICON_PATH):
    with open(LEXICON_PATH, 'rb') as f:
        lexicon = pickle.load(f)
    print('lexicon loaded')
else:
    lexicon = create_lexicon(data_x)
    with open(LEXICON_PATH, 'wb') as f:
        pickle.dump(lexicon, f)
    print('lexicon created')


print('LEX: ', lexicon[:5])
# DICTIONARIES
# If dictionaries file exists, load it
# else, create new dictionaries and save them to the file
if os.path.exists(DICTIONARIES_PATH):
    with open(DICTIONARIES_PATH, 'rb') as f:
        dictionary, reverse_dictionary = pickle.load(f)
    print('dictionaries loaded')
else:
    dictionaries = create_dictionary(lexicon)
    dictionary, reverse_dictionary = dictionaries
    with open(DICTIONARIES_PATH, 'wb') as f:
        pickle.dump(dictionaries, f)
    print('dictionaries created')

# DATA X RAW
# If features file exists, load it
# else, process new features and save it to the file
if os.path.exists(DATA_X_PATH):
    with open(DATA_X_PATH, 'rb') as f:
        data_x = pickle.load(f)
    print('features loaded')
else:
    data_x = convert_sentences(data_x, dictionary)
    with open(DATA_X_PATH, 'wb') as f:
        pickle.dump(data_x, f)
    print('features processed')

# DATA X FOR EMBEDDINGS
# If embedding data file exists, load it
# else, process new embedding data and save it to the file
if os.path.exists(EMBED_DATA_X_PATH):
    with open(EMBED_DATA_X_PATH, 'rb') as f:
        embedding_data = pickle.load(f)
    print('embedding data loaded')
else:
    # flatten and remove neighbour zeros
    embedding_data = np.reshape(data_x, (-1))
    embedding_data = [x for i, x in enumerate(embedding_data) if sum(embedding_data[i - 1:i]) > 0]
    with open(EMBED_DATA_X_PATH, 'wb') as f:
        pickle.dump(embedding_data, f)
    print('embedding data created')

# EMBEDDINGS
# If embeddings file exists, load it
# else, create new embeddings and save it to the file
if os.path.exists(EMBEDDINGS_PATH):
    with open(EMBEDDINGS_PATH, 'rb') as f:
        embeddings = pickle.load(f)
    print('embeddings loaded')
else:
    embeddings = train_embeddings(embedding_data, len(lexicon))
    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(embeddings, f)
    print('embeddings processed')

print('Lexicon length: ', len(lexicon))
embedding_to_visualize = []
labels = []
for i, label in enumerate(lexicon):
    labels.append(label)
    embedding_to_visualize.append(embeddings[i])
    if i > 8000:
        break

embedding_to_visualize = np.array(embedding_to_visualize)
visualize_embeddings(embedding_to_visualize, labels)
print('embeddings visualised in tensorboard')
os.system('tensorboard --logdir=C:\dev\ToxicityClassification\log')
