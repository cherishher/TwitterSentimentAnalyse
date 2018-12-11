import re
import string

import nltk
import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer

from preprocess.mapdata import CONTRACTION_MAP


# input: a 100-d vector
# return: -1 for negative, 1 for positive
def my_predict(vec):
    M = len(vec)
    vec = np.array(vec).reshape(1, M)

    x_data = tf.placeholder(shape=[1, M], dtype=tf.float32)

    A = tf.Variable(tf.random_normal(shape=[M, 1]))
    b = tf.Variable(tf.random_normal(shape=[1, 1]))

    model_output = tf.subtract(tf.matmul(x_data, A), b)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, '../model/MoModel')
        result = sess.run(model_output, feed_dict={x_data: vec})

    tf.reset_default_graph()

    return result[0][0]


# token
def tokenization(text):
    sentences = nltk.sent_tokenize(text)
    tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
    return tokens


# remove symbols
def rmchar(text):
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    text = filter(None, [pattern.sub('', token) for token in text])
    return text


# remove repeat character
def rmrepeat(text):
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    match_substitution = r'\1\2\3'

    def replace(old_word):
        if wordnet.synsets(old_word):
            return old_word
        new_word = repeat_pattern.sub(match_substitution, old_word)
        return replace(new_word) if new_word != old_word else new_word

    text = [replace(word) for word in text]
    return text


# lemm
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


# can't -> can not follow the map
def expand(sentence, contraction_mapping):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_sentence = contractions_pattern.sub(expand_match, sentence)
    return expanded_sentence


# stop words
stopword = stopwords.words('english')
custom_word = ['']
for w in custom_word:
    stopword.append(w)

# load word-vectors
model = Word2Vec.load("../model/word2vec.model")

# load high-chi2 words
top_k = 10000
high_chi2_words = []
fr = open("../data/chi2_rank.txt", "r")
rank = 0
for line in fr:
    rank += 1
    if rank > top_k:
        break
    high_chi2_words.append(line.strip())
fr.close()


def sentiment_analysing(T):
    T = [T.strip()]

    # Expand contraction
    expandedtext = [expand(sentence, CONTRACTION_MAP) for sentence in T]
    # remove symbols
    T = rmchar(expandedtext)
    T = " ".join(T)
    T = re.split(r'\W+', T)  # remove @#$%^&*
    T = rmrepeat(T)  # remove repeat stilllllll -> still

    # delete ''
    T = filter(None, T)

    tagged_sent = pos_tag(T)

    # lemmatizer
    wnl = WordNetLemmatizer()
    wordnet_lemmatizer = WordNetLemmatizer()
    T = []
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        T.append(wnl.lemmatize(tag[0], pos=wordnet_pos))

    # Stemmer
    stemmer = nltk.stem.PorterStemmer()
    stem_1 = [stemmer.stem(t) for t in T]

    T = " ".join(T)  # list -> str

    T = T.lower()  # lowercase
    T = T.encode('utf-8')  # remove u'...'

    # token
    T = WordPunctTokenizer().tokenize(T)  # token

    T = [word for word in T if word not in stopword]

    vec = [0.0] * 100
    word_cnt = 0
    for word in T:
        if word in high_chi2_words and word in model:
            for i in range(0, 100):
                vec[i] += model[word][i]
            word_cnt += 1
    for i in range(0, 100):
        if word_cnt != 0:
            vec[i] = vec[i] / word_cnt

    result = my_predict(vec)

    if result > 0:
        return ('positive')
    else:
        return ('negative')
