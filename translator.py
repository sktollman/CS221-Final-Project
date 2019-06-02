import argparse
from collections import namedtuple
from cmd import Cmd
import csv
import nltk
nltk.download('wordnet', quiet=True)
from nltk.corpus import wordnet
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import shakespeare_fluency
import language_model
import synonyms
import search_util

# from https://www.kaggle.com/emmabel/word-occurrences-in-shakespeare
FILENAME = 'shakespeare.csv'

bigram_model = shakespeare_fluency.shakespeare_bigram_model()

def read_file():
    with open(FILENAME) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        word_counts = {row[0].rstrip(): int(row[1]) for row in csv_reader}
        return word_counts

word_counts = read_file()

class TranslationProblem(search_util.SearchProblem):
    def __init__(self, query, synonymGenerator, costFn):
        self.query = query.split()
        self.synonymGenerator = synonymGenerator
        self.costFn = costFn

    def startState(self):
        return (shakespeare_fluency.SENTENCE_BEGIN,)

    def isEnd(self, state):
        return len(state) == (len(self.query) + 1)

    def succAndCost(self, state):
        nextWord = self.query[len(state) - 1]
        synonyms = self.synonymGenerator(nextWord)

        # print(state)
        # print(nextWord)

        results = []
        for s in synonyms:
            newState = list(state)
            newState.append(s)
            results.append((s, tuple(newState), self.costFn(newState)))

        return results

def translate(query, synonymGenerator, scorer):
    if len(query) == 0:
        return ''

    ucs = search_util.UniformCostSearch(verbose=0)
    ucs.solve(TranslationProblem(query, synonymGenerator, scorer))

    return ' '.join(list(ucs.actions))

def nltk_synonmys(word):
    # find all ntlk synonyms
    synonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())

    # now find those that appear in shakespeare
    shakespeare_synonyms = [w for w in word_counts if w in synonyms]
    # if no matches in shakespearean text, use original word
    if not shakespeare_synonyms: shakespeare_synonyms = [word]
    return shakespeare_synonyms

cache = dict()

def bigram_sentence_fluency(words):
    if len(words) <= 1: return 0
    words = tuple(words)
    if words in cache: return cache[words]
    result = bigram_model(words[0], words[1]) + bigram_sentence_fluency(words[2:])
    cache[words] = result
    return result

synonym_generators = {'NLTK': nltk_synonmys,
    'Word Vector': synonyms.shakes_synonym}

sentence_scorers = {'Bigram': bigram_sentence_fluency, 'Language Model':
    lambda words: language_model.score_sentence(' '.join(words[1:]))}

def run_models(sentence, oracle):
    for syn_name, synonym_generator in synonym_generators.items():
        print('{} synonyms:'.format(syn_name))
        # unigram frequency model
        syn = lambda word: synonym_generator(word)[0]
        unigram = ' '.join(map(syn, sentence.split()))
        print('Unigram frequency model: {}'.format(unigram))
        # score = sentence_bleu([oracle.split()], unigram.split(),
        #     smoothing_function=SmoothingFunction().method1)
        # print('Bleu score: {}'.format(round(score, 4)))

        for name, scorer in sentence_scorers.items():
            res = translate(s, synonym_generator, scorer)
            print('{} translation: {}'.format(name, res))
            score = language_model.score_sentence(res)
            print('LSTM model score: {}'.format(score))
        print()

SENTENCES = [
    """You agree now that we’re not imagining this, don’t you""",
    """I’ll meet it if it’s the last thing I do""",
    """That’s why I’ve begged him to come on our shift tonight"""
]

ORIGINALS = [
    """Is not this something more than fantasy?""",
    """I’ll cross it though it blast me.""",
    """Therefore I have entreated him along""",
    """With us to watch the minutes of this night"""
]

class InteractiveTranslation(Cmd):
    prompt = '> '
    intro = "Welcome! Enter a sentence to translate into Shakespeare"

    def default(self, sentence):
        run_models(sentence)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shell', action='store_true')
    args = parser.parse_args()

    if args.shell:
        InteractiveTranslation().cmdloop()
    else: # run baseline tests
        for s, o in zip(SENTENCES, ORIGINALS):
            print('No Fear: {}'.format(s))
            print('Original: {}'.format(o))
            run_models(s, o)
            print() # newline
