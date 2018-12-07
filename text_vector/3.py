import pickle
import sys

from gensim.models import Word2Vec

top_k = 10000
if len(sys.argv) != 2:
    print("Usage: python 3.py [k: top k words]")
    sys.exit(0)
else:
    top_k = int(sys.argv[1])

model = Word2Vec.load("word2vec.model")
print(len(model.wv.vocab.keys()))

# load high-chi2 words

high_chi2_words = []

fr = open("chi2_rank2.txt", "r")
rank = 0
for line in fr:
    rank += 1
    if rank > top_k:
        break
    high_chi2_words.append(line.strip())
fr.close()

# print high_chi2_words
print(len(high_chi2_words))

# load text data

load_file = open('data.dat', 'rb')
L = pickle.load(load_file)
load_file.close()

# 1600000 elements for L
print(L[1])
print("===============Strat==============")

fo = open("text_vector_" + str(top_k) + ".txt", "w")

# construct the word-count dictionary
word_dict = {}
total0 = 0
total1 = 0
time = 0

for lable_text in L:
    print(time)

    lable = int(lable_text[0][0])
    text = lable_text[1]

    fo.write(str(time) + " ")
    vec = [0.0] * 100
    word_cnt = 0
    for word in text:
        if word in high_chi2_words and word in model:
            for i in range(0, 100):
                vec[i] += model[word][i]
            word_cnt += 1
    for i in range(0, 100):
        if word_cnt != 0:
            vec[i] = vec[i] / word_cnt
        fo.write(str(vec[i]) + ' ')
    fo.write('\n')

    time += 1

fo.close()
