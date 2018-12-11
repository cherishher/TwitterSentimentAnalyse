# Usage: python gen_text_vector_for_test.py [k]
# Input: data.dat word2vec.model chi2_rank.txt
# Output: test_text_vect_k.txt

'''
The program is to convert some random testing text into a vector.

'''

from gensim.models import Word2Vec
from gensim.models import word2vec
import pickle
import sys
import random

top_k = 10000

model = Word2Vec.load("../model/word2vec.model")
print(len(model.wv.vocab.keys()))

# load high-chi2 words

high_chi2_words = []

fr = open("../data/chi2_rank.txt", "r")
rank = 0
for line in fr:
    rank += 1
    if rank > top_k:
        break
    high_chi2_words.append(line.strip())
fr.close()

# load text data

load_file = open('../data/newdatastop.dat', 'rb')
L = pickle.load(load_file)
load_file.close()

# 1600000 elements for L
print("===============Strat==============")

fo = open("../data/test_text_vector.txt", "w")
fo1 = open("../data/test_text_label.txt", "w")

# construct the word-count dictionary
word_dict = {}
total0 = 0
total1 = 0
time = 0

for lable_text in L:
    if random.random() < 0.05:
        lable = int(lable_text[0][0])
        text = lable_text[1]

        fo.write(str(time) + " ")
        fo1.write(str(time) + ' ' + str(lable) + '\n')
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

fo.close()
fo1.close()
