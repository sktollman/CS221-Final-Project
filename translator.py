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

def find_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
    return synonyms

def all_shakespeare_synonmys(word):
    synonyms = find_synonyms(word)
    shakespeare_synonyms = [w for w in word_counts if w in synonyms]
    # if no matches in shakespearean text, use original word
    if not shakespeare_synonyms: shakespeare_synonyms = [word]
    return shakespeare_synonyms

def shakespeare_synonym(word):
    shakespeare_synonyms = sorted(all_shakespeare_synonmys(word),
        key=lambda w: word_counts[w] if w in word_counts else 0, reverse=True)

    # if the top match is the same as the original word, use the second best
    # match
    if shakespeare_synonyms[0] == word and len(shakespeare_synonyms) > 1:
        return shakespeare_synonyms[1]

    return shakespeare_synonyms[0]

cache = dict()

def sentence_fluency(words):
    if len(words) <= 1: return 0
    words = tuple(words)
    if words in cache: return cache[words]
    result = bigram_model(words[0], words[1]) + sentence_fluency(words[2:])
    cache[words] = result
    return result

def translate_to_shakespeare(sentence, synonym_generator):
    words = sentence.split()
    possibilities = [synonym_generator(w) for w in words]

    possible_sentences = []
    def populate(curr, remaining):
        if not remaining:
            possible_sentences.append(curr)
            return

        for x in remaining[0]:
            y = list(curr)
            y.append(x)
            populate(y, remaining[1:])

    populate([shakespeare_fluency.SENTENCE_BEGIN], possibilities)

    m = max(possible_sentences, key=sentence_fluency)
    return ' '.join(m[1:])

def language_model_translation(sentence, synonym_generator):
    words = sentence.split()
    possibilities = [synonym_generator(w) for w in words]

    possible_sentences = []
    def populate(curr, remaining):
        if not remaining:
            possible_sentences.append(curr)
            return

        for x in remaining[0]:
            y = list(curr)
            y.append(x)
            populate(y, remaining[1:])

    populate([shakespeare_fluency.SENTENCE_BEGIN], possibilities)

    # for s in possible_sentences:
    #     print(language_model.score_sentence(' '.join(s[1:])))

    m = max(possible_sentences,
        key=lambda words: language_model.score_sentence(' '.join(words[1:])))
    return ' '.join(m[1:])

def run_models(sentence, oracle, synonym_generator):
     # unigram frequency model
    syn = lambda word: synonym_generator(word)[0]
    unigram = ' '.join(map(syn, sentence.split()))
    print('Unigram frequency model: {}'.format(unigram))
    # score = sentence_bleu([oracle.split()], unigram.split(),
    #     smoothing_function=SmoothingFunction().method1)
    # print('Bleu score: {}'.format(round(score, 4)))

    # unigram model + bigram sentence fluency
    fluency = translate_to_shakespeare(sentence, synonym_generator)
    print('Unigram model + bigram sentence fluency: {}'.format(fluency))
    # score = sentence_bleu([oracle.split()], fluency.split(),
    #     smoothing_function=SmoothingFunction().method1)
    # print('Bleu score: {}'.format(round(score,4)))

    lm = language_model_translation(sentence, synonym_generator)
    print('Unigram model + language model scores: {}'.format(lm))
    print('LSTM model score: {}'.format(language_model.score_sentence(lm)))

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

            print('NLTK synoynms:')
            res = translate(s, all_shakespeare_synonmys, sentence_fluency)
            print('Bigram translation: {}'.format(res))
            res = translate(s, all_shakespeare_synonmys,
                lambda words: language_model.score_sentence(' '.join(words[1:])))
            print('Language model translation: {}'.format(res))

            print('Word vector synoynms:')
            res = translate(s, synonyms.shakes_synonym, sentence_fluency)
            print('Bigram translation: {}'.format(res))
            res = translate(s, synonyms.shakes_synonym,
                lambda words: language_model.score_sentence(' '.join(words[1:])))
            print('Language model translation: {}'.format(res))
            print()

            # print('Using NLTK synonyms:')
            # run_models(s, o, all_shakespeare_synonmys)
            # print('Using word vector synoynms:')
            # run_models(s, o, synonyms.shakes_synonym)
            # print() # newline
