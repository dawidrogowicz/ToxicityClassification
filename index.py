import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle
import os
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
from embeddings import train_embeddings, plot_embeddings
from collections import Counter

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
    _lexicon = [count[0] for count in Counter(_lexicon).most_common(n_tokens)]
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


def create_dictionary(sentences, _lexicon):
    _dictionary = dict()
    for entry in _lexicon:
        _dictionary[entry] = len(_dictionary) + 1
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


# def generate_batch(x, y, batch_size=256):
#     assert len(x) == len(y)
#     assert len(np.shape(x)) == 2 and len(np.shape(y)) == 2
#
#     data = list()
#
#     while len(data) < len(x):


# START
df = pd.read_csv(TRAIN_PATH)
# limit number of samples until model is ready
df = df[:10000]

data_x = df['comment_text']
data_y = np.array(df.iloc[:, 2:])

del df

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

# If dictionaries file exists, load it
# else, create new dictionaries and save them to the file
if os.path.exists(DICTIONARIES_PATH):
    with open(DICTIONARIES_PATH, 'rb') as f:
        dictionary, reverse_dictionary = pickle.load(f)
    print('dictionaries loaded')
else:
    dictionaries = create_dictionary(data_x, lexicon)
    dictionary, reverse_dictionary = dictionaries
    with open(DICTIONARIES_PATH, 'wb') as f:
        pickle.dump(dictionaries, f)
    print('dictionaries created')

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

# If embeddings file exists, load it
# else, create new embeddings and save it to the file
if os.path.exists(EMBEDDINGS_PATH):
    with open(EMBEDDINGS_PATH, 'rb') as f:
        embeddings = pickle.load(f)
    print('embeddings loaded')
else:
    vocabulary_size = len(lexicon)
    lexicon = [dictionary.get(x, 0) for x in lexicon]
    # flatten and remove neighbour zeros
    embedding_data = [x for i, x in enumerate(lexicon) if sum(lexicon[i - 1:i]) > 0]
    embeddings = train_embeddings(embedding_data, vocabulary_size, reverse_dictionary)
    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(embeddings, f)
    print('embeddings processed')


tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
plot_only = 400
embeddings_2d = tsne.fit_transform(embeddings[:plot_only, :])
labels = [reverse_dictionary.get(i, 0) for i in range(1, plot_only + 1)]

plot_embeddings(embeddings_2d, labels, 'plot2d.png')

# train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=.3, random_state=1)
# length = len(lexicon)
#
# model = model((None, length))
# model.fit(train_x, train_y, 10)
#
# acc = model.evaluate(test_x, test_y)
# print(acc)
