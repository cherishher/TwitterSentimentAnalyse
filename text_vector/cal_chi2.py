# Usage: python cal_chi2.py
# Input: data.dat
# Output: chi2_rank.txt text_label.txt

'''
This program read the label-text data, use it to calculate chi-2 value
for each word, and finally output a ranked list of words based on chi-2.

By the way, it also output a text_label.txt, which contains text-label pairs.
This is just for usage of classfication step.

'''

import pickle

# read label-text data
load_file = open('data.dat', 'rb')
L = pickle.load(load_file)
load_file.close()

# Selection part
print
print
"===============Strat Selection=============="

fo1 = open("text_label.txt", "w")

# construct the word-count dictionary:
# {"word": [negative cnt, positivecnt, chi-2], ...}
word_dict = {}
total0 = 0
total1 = 0
time = 0

for lable_text in L:
    print "#time=", time
    lable = int(lable_text[0][0])
    if lable == 4:
        lable = 1
    else:
        lable = -1
    text = lable_text[1]

    fo1.write(str(time) + ' ' + str(lable) + '\n')  # output the text_label.txt

    # count the appearance rate of each word
    if lable == -1:
        total0 += 1
        for word in text:
            if word in word_dict:
                word_dict[word][0] += 1
            else:
                word_dict[word] = [1, 0, 0]
    else:
        total1 += 1
        for word in text:
            if word in word_dict:
                word_dict[word][1] += 1
            else:
                word_dict[word] = [0, 1, 0]
    time += 1
fo1.close()

# calculate chi2 for each word and select the top-xxxx
fo2 = open("chi2_rank.txt", "w")

for word in word_dict:
    A = word_dict[word][0]
    B = word_dict[word][1]
    C = total0 - A
    D = total1 - B
    word_dict[word][2] = (A * D - B * C) * (A * D - B * C) / float(A + B) / float(C + D)  # compute chi-2

# sort the word_dict
rank = 0

word_dict = sorted(word_dict.items(), key=lambda x: x[1][2], reverse=True)
for ele in word_dict:
    rank += 1
    print(ele[0], ele[1])
    fo2.write(str(ele[0]) + '\n')  # Output the chi2_rank.txt

fo2.close()
