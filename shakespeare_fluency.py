import collections
import math

SENTENCE_BEGIN = '-BEGIN-'

def sliding(xs, windowSize):
    for i in range(1, len(xs) + 1):
        yield xs[max(0, i - windowSize):i]

def removeAll(s, chars):
    return ''.join(filter(lambda c: c not in chars, s))

def alphaOnly(s):
    s = s.replace('-', ' ')
    return str(filter(lambda c: c.isalpha() or c == ' ', s))

def cleanLine(l):
    return alphaOnly(l.strip().lower())

def words(l):
    return l.split()

# Make an n-gram model of words in text from a corpus.
# Taken from CS221 assignment starter code
def makeLanguageModels(path):
    unigramCounts = collections.Counter()
    totalCounts = 0
    bigramCounts = collections.Counter()
    bitotalCounts = collections.Counter()
    VOCAB_SIZE = 600000
    LONG_WORD_THRESHOLD = 5
    LENGTH_DISCOUNT = 0.15

    def bigramWindow(win):
        assert len(win) in [1, 2]
        if len(win) == 1:
            return (SENTENCE_BEGIN, win[0])
        else:
            return tuple(win)

    with open(path, 'r') as f:
        for l in f:
            ws = words(cleanLine(l))
            unigrams = [x[0] for x in sliding(ws, 1)]
            bigrams = [bigramWindow(x) for x in sliding(ws, 2)]
            totalCounts += len(unigrams)
            unigramCounts.update(unigrams)
            bigramCounts.update(bigrams)
            bitotalCounts.update([x[0] for x in bigrams])

    def unigramCost(x):
        if x not in unigramCounts:
            length = max(LONG_WORD_THRESHOLD, len(x))
            return -(length * math.log(LENGTH_DISCOUNT) + math.log(1.0) - math.log(VOCAB_SIZE))
        else:
            return math.log(totalCounts) - math.log(unigramCounts[x])

    def bigramModel(a, b):
        return math.log(bitotalCounts[a] + VOCAB_SIZE) - math.log(bigramCounts[(a, b)] + 1)

    return unigramCost, bigramModel

# from https://www.kaggle.com/kingburrito666/shakespeare-plays/downloads/shakespeare-plays.zip/4
CORPUS = 'alllines.txt'

def shakespeare_bigram_model():
    return makeLanguageModels(CORPUS)[1]
