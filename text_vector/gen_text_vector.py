# Usage: python gen_text_vector.py [k]
# Input: data.dat word2vec.model chi2_rank.txt
# Output: text_vect_k.txt

'''
The program is to convert each text into a vector.

'''

from gensim.models import Word2Vec
from gensim.models import word2vec
import pickle
import sys

top_k=10000
if len(sys.argv)!= 2:
	print "Usage: python gen_text_vector.py [k: top k words]"
	sys.exit(0)
else:
	top_k = int(sys.argv[1])

# load word-vectors
model = Word2Vec.load("word2vec.model")
#print(len(model.wv.vocab.keys()))

# load chi2-ranked words
high_chi2_words=[]

fr = open("chi2_rank.txt","r")
rank = 0
for line in fr:
	rank+=1
	if rank > top_k:
		break 
	high_chi2_words.append(line.strip())
fr.close()


# load text data
load_file = open('data.dat','rb')
L = pickle.load(load_file) # 1600000 elements for L
load_file.close()



# convert each text into a vector
print
print "===============Strat=============="

fo=open("text_vector_"+str(top_k)+".txt","w")

time=0
for lable_text in L:
	print time

	lable = int(lable_text[0][0])
	text = lable_text[1]

	fo.write(str(time)+" ")
	vec=[0.0]*100
	word_cnt=0
	for word in text:
		# if the word is rank-k word and also in word2vec.model
		# add its vector to the text-vector
		if word in high_chi2_words and word in model:
			for i in range(0,100):
				vec[i]+=model[word][i]
			word_cnt+=1
	# calculate the average
	for i in range(0,100):
		if word_cnt != 0:
			vec[i]=vec[i]/word_cnt
		fo.write(str(vec[i])+' ')
	fo.write('\n')
	time+=1

fo.close()
