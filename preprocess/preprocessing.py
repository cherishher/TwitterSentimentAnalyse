import pickle
import re
import csv
import nltk
import string
from nltk import word_tokenize, pos_tag
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from preprocess.mapdata import CONTRACTION_MAP

wordnet_lemmatizer = WordNetLemmatizer()

target = []
text = []
with open('data.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        target += [row[0]]
        text += [row[5]]

L = [[0 for j in range(2)] for i in range(1600000)]


# T = list([text[1620]])   # 270  # 1620

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

for i in range(1600000):

    T = list([text[i]])
    # print(T)

    # Expand contraction
    expandedtext = [expand(sentence, CONTRACTION_MAP) for sentence in T]
    # remove symbols
    T = rmchar(expandedtext)
    T = " ".join(T)
    T = re.split(r'\W+', T)  # remove @#$%^&*
    # print(T)
    T = rmrepeat(T)  # remove repeat stilllllll -> still

    T = filter(None, T)  # delete ''

    tagged_sent = pos_tag(T)
    # lemmatizer
    wnl = WordNetLemmatizer()
    T = []
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        T.append(wnl.lemmatize(tag[0], pos=wordnet_pos))

    # T = [snowball_stemmer.stem(t) for t in T]
    # # print(T)
    # Stemmer
    stemmer = nltk.stem.PorterStemmer()
    stem_1 = [stemmer.stem(t) for t in T]

    T = " ".join(T)  # list -> str
    # lower case
    T = T.lower()  # lowercase
    T = T.encode('utf-8')  # remove u'...'

    # token
    T = WordPunctTokenizer().tokenize(T)  # token

    T = [word for word in T if word not in stopword]

    text[i] = T
    target[i] = list(target[i])
    L[i] = [target[i], text[i]]

    print(i)

save_file = open('/Users/Cindy/Desktop/newdatastop.dat', 'wb')
pickle.dump(L, save_file)
save_file.close()
