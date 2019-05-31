import argparse
from collections import namedtuple
from cmd import Cmd
import csv
import nltk
nltk.download('wordnet', quiet=True)
from nltk.corpus import wordnet
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import shakespeare_fluency

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

def translate_to_shakespeare(sentence):
    words = sentence.split()
    possibilities = [all_shakespeare_synonmys(w) for w in words]

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

def run_models(sentence, oracle):
     # unigram frequency model
    unigram = ' '.join(map(shakespeare_synonym, sentence.split()))
    print('Unigram frequency model: {}'.format(unigram))
    score = sentence_bleu([oracle.split()], unigram.split(),
        smoothing_function=SmoothingFunction().method1)
    print('Bleu score: {}'.format(round(score, 4)))

    # unigram model + bigram sentence fluency
    fluency = translate_to_shakespeare(sentence)
    print('Unigram model + bigram sentence fluency: {}'.format(fluency))
    score = sentence_bleu([oracle.split()], fluency.split(),
        smoothing_function=SmoothingFunction().method1)
    print('Bleu score: {}'.format(round(score,4)))


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
