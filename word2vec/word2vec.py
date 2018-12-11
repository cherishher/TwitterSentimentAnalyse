import argparse
import math
import struct
import sys
import time
import warnings

import numpy as np

from multiprocessing import Pool, Value, Array

class VocabItem:
    def __init__(self, word):
        self.word = word
        self.count = 0

class Vocab:
    def __init__(self, fi, min_count):
        vocab_items = []
        vocab_hash = {}
        word_count = 0
        fi = open(fi, 'r')

        for token in ['<bol>', '<eol>']:
            vocab_hash[token] = len(vocab_items)
            vocab_items.append(VocabItem(token))

        for line in fi:
            tokens = line.split()
            for token in tokens:
                if token not in vocab_hash:
                    vocab_hash[token] = len(vocab_items)
                    vocab_items.append(VocabItem(token))

                vocab_items[vocab_hash[token]].count += 1
                word_count += 1
            
                if word_count % 10000 == 0:
                    sys.stdout.write("\rReading word %d" % word_count)
                    sys.stdout.flush()

            vocab_items[vocab_hash['<bol>']].count += 1
            vocab_items[vocab_hash['<eol>']].count += 1
            word_count += 2

        self.bytes = fi.tell()
        self.vocab_items = vocab_items
        self.vocab_hash = vocab_hash
        self.word_count = word_count

        self.__sort(min_count)

        print 'Total words in training file: %d' % self.word_count
        print 'Total bytes in training file: %d' % self.bytes
        print 'Vocab size: %d' % len(self)

    def __getitem__(self, i):
        return self.vocab_items[i]

    def __len__(self):
        return len(self.vocab_items)

    def __iter__(self):
        return iter(self.vocab_items)

    def __contains__(self, key):
        return key in self.vocab_hash

    def __sort(self, min_count):
        tmp = []
        tmp.append(VocabItem('<unk>'))
        unk_hash = 0
        
        count_unk = 0
        for token in self.vocab_items:
            if token.count < min_count:
                count_unk += 1
                tmp[unk_hash].count += token.count
            else:
                tmp.append(token)

        tmp.sort(key=lambda token : token.count, reverse=True)

        vocab_hash = {}
        for i, token in enumerate(tmp):
            vocab_hash[token.word] = i

        self.vocab_items = tmp
        self.vocab_hash = vocab_hash

        print
        print 'Unknown vocab size:', count_unk

    def indices(self, tokens):
        return [self.vocab_hash[token] if token in self else self.vocab_hash['<unk>'] for token in tokens]


class UnigramTable:

    def __init__(self, vocab):
        vocab_size = len(vocab)
        power = 0.75
        norm = sum([math.pow(t.count, power) for t in vocab]) # Normalizing constant

        table_size = 1e9
        table = np.zeros(int(table_size), dtype=np.uint32)

        print 'Filling unigram table'
        p = 0
        i = 0
        for j, unigram in enumerate(vocab):
            p += float(math.pow(unigram.count, power))/norm
            while i < table_size and float(i) / table_size < p:
                table[i] = j
                i += 1
        self.table = table

    def sample(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]

def sigmoid(z):
    if z > 6:
        return 1.0
    elif z < -6:
        return 0.0
    else:
        return 1 / (1 + math.exp(-z))

def init_net(dim, vocab_size):
    tmp = np.random.uniform(low=-0.5/dim, high=0.5/dim, size=(vocab_size, dim))
    syn0 = np.ctypeslib.as_ctypes(tmp)
    syn0 = Array(syn0._type_, syn0, lock=False)

    # Init syn1 with zeros
    tmp = np.zeros(shape=(vocab_size, dim))
    syn1 = np.ctypeslib.as_ctypes(tmp)
    syn1 = Array(syn1._type_, syn1, lock=False)

    return (syn0, syn1)

def train_process(pid):

    start = vocab.bytes / num_processes * pid
    end = vocab.bytes if pid == num_processes - 1 else vocab.bytes / num_processes * (pid + 1)
    for i in range(epoch):
        fi.seek(start)


        alpha = starting_alpha

        word_count = 0
        last_word_count = 0

        while fi.tell() < end:
            line = fi.readline().strip()

            if not line:
                continue

            sent = vocab.indices(['<bol>'] + line.split() + ['<eol>'])

            for sent_pos, token in enumerate(sent):
                if word_count % 10000 == 0:
                    global_word_count.value += (word_count - last_word_count)
                    last_word_count = word_count

                    alpha = starting_alpha * (1 - float(global_word_count.value) / vocab.word_count)
                    if alpha < starting_alpha * 0.0001: alpha = starting_alpha * 0.0001

                    sys.stdout.write("\rAlpha: %f Progress: %d of %d (%.2f%%)" %
                                     (alpha, global_word_count.value, vocab.word_count,
                                      float(global_word_count.value) / vocab.word_count * 100))
                    sys.stdout.flush()

                current_win = np.random.randint(low=1, high=win + 1)
                context_start = max(sent_pos - current_win, 0)
                context_end = min(sent_pos + current_win + 1, len(sent))
                context = sent[context_start:sent_pos] + sent[sent_pos + 1:context_end]  # Turn into an iterator?

                # CBOW
                if cbow:
                    neu1 = np.mean(np.array([syn0[c] for c in context]), axis=0)
                    assert len(neu1) == dim, 'neu1 and dim do not agree'

                    neu1e = np.zeros(dim)

                    classifiers = [(token, 1)] + [(target, 0) for target in table.sample(neg)]

                    for target, label in classifiers:
                        z = np.dot(neu1, syn1[target])
                        p = sigmoid(z)
                        g = alpha * (label - p)
                        neu1e += g * syn1[target]
                        syn1[target] += g * neu1

                    for context_word in context:
                        syn0[context_word] += neu1e

                # Skip-gram
                else:
                    for context_word in context:
                        neu1e = np.zeros(dim)

                        classifiers = [(token, 1)] + [(target, 0) for target in table.sample(neg)]
                        for target, label in classifiers:
                            z = np.dot(syn0[context_word], syn1[target])
                            p = sigmoid(z)
                            g = alpha * (label - p)
                            neu1e += g * syn1[target]
                            syn1[target] += g * syn0[context_word]


                        syn0[context_word] += neu1e

                word_count += 1

    global_word_count.value += (word_count - last_word_count)
    sys.stdout.write("\rAlpha: %f Progress: %d of %d (%.2f%%)" %
                     (alpha, global_word_count.value, vocab.word_count,
                      float(global_word_count.value)/vocab.word_count * 100))
    sys.stdout.flush()
    fi.close()

def save(vocab, syn0, fo):
    print 'Saving model to', fo
    dim = len(syn0[0])

    fo = open(fo, 'w')
    fo.write('%d %d\n' % (len(syn0), dim))
    for token, vector in zip(vocab, syn0):
        word = token.word
        vector_str = ' '.join([str(s) for s in vector])
        fo.write('%s %s\n' % (word, vector_str))

    fo.close()

def __init_process(*args):
    global vocab, syn0, syn1, table, cbow, neg, dim, starting_alpha, epoch
    global win, num_processes, global_word_count, fi
    
    vocab, syn0_tmp, syn1_tmp, table, cbow, neg, dim, starting_alpha, win, num_processes, global_word_count, epoch = args[:-1]
    fi = open(args[-1], 'r')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        syn0 = np.ctypeslib.as_array(syn0_tmp)
        syn1 = np.ctypeslib.as_array(syn1_tmp)

def train(fi, fo, cbow, neg, dim, alpha, win, min_count, num_processes, epoch):
    vocab = Vocab(fi, min_count)

    syn0, syn1 = init_net(dim, len(vocab))

    global_word_count = Value('i', 0)
    print 'Initializing unigram table'
    table = UnigramTable(vocab)
    t0 = time.time()
    pool = Pool(processes=num_processes, initializer=__init_process,
                initargs=(vocab, syn0, syn1, table, cbow, neg, dim, alpha,
                          win, num_processes, global_word_count, epoch, fi))
    pool.map(train_process, range(num_processes))
    t1 = time.time()
    print
    print 'Completed training. Training took', (t1 - t0) / 60, 'minutes'

    # Save model to file
    save(vocab, syn0, fo)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', help='Training file', dest='fi', required=True)
    parser.add_argument('-model', help='Output model file', dest='fo', required=True)
    parser.add_argument('-cbow', help='1 for CBOW, 0 for skip-gram', dest='cbow', default=1, type=int)
    parser.add_argument('-negative', help='Number of negative examples (>0) for negative sampling', dest='neg', default=5, type=int)
    parser.add_argument('-dim', help='Dimensionality of word embeddings', dest='dim', default=100, type=int)
    parser.add_argument('-alpha', help='Starting alpha', dest='alpha', default=0.025, type=float)
    parser.add_argument('-window', help='Max window length', dest='win', default=5, type=int) 
    parser.add_argument('-min-count', help='Min count for words used to learn <unk>', dest='min_count', default=5, type=int)
    parser.add_argument('-processes', help='Number of processes', dest='num_processes', default=1, type=int)
    parser.add_argument('-epoch', help='Number of training epochs', dest='epoch', default=1, type=int)
    args = parser.parse_args()

    train(args.fi, args.fo, bool(args.cbow), args.neg, args.dim, args.alpha, args.win,
          args.min_count, args.num_processes,args.epoch)
