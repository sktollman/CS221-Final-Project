from collections import namedtuple
import csv
import nltk
nltk.download('wordnet', quiet=True)
from nltk.corpus import wordnet
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

def shakespeare_synonym(word):
    synonyms = find_synonyms(word)
    shakespeare_synonyms = \
        [(w, c) for w, c in word_counts.items() if w in synonyms]

    # if no matches in shakespearean text, use original word
    if not shakespeare_synonyms: return word

    shakespeare_synonyms = \
        sorted(shakespeare_synonyms, key=lambda x: x[1], reverse=True)

    # if the top match is the same as the original word, use the second best
    # match
    if shakespeare_synonyms[0][0] == word and len(shakespeare_synonyms) > 1:
        return shakespeare_synonyms[1][0]

    return shakespeare_synonyms[0][0]

def all_shakespeare_synonmys(word):
    synonyms = find_synonyms(word)
    shakespeare_synonyms = \
        [w for w, c in word_counts.items() if w in synonyms]
    # if no matches in shakespearean text, use original word
    if not shakespeare_synonyms: shakespeare_synonyms = [word]
    return shakespeare_synonyms

def sentence_fluency(words):
    result = bigram_model(shakespeare_fluency.SENTENCE_BEGIN, words[0])
    for i in range(len(words) - 1):
        result += bigram_model(words[i], words[i+1])
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

    populate([], possibilities)

    # print(possible_sentences)
    m = max(possible_sentences, key=sentence_fluency)
    # print(m)
    return ' '.join(m)

SENTENCES = [
"""
Horatio says we’re imagining it,
and won’t let himself believe anything about
this horrible thing that we’ve seen twice now.
That’s why I’ve begged him
to come on our shift tonight,
so that if the ghost appears
he can see what we see and speak to it.
""",
"""
What’s going on, Horatio? You’re pale and trembling.
You agree now that we’re not imagining this, don’t you?
What do you think about it?
""",
"""
Wait, look! It has come again.
I’ll meet it if it’s the last thing I do. —Stay here, you hallucination!
"""
]

ORIGINALS = [
"""
Horatio says ‘tis but our fantasy
And will not let belief take hold of him
Touching this dreaded sight twice seen of us.
Therefore I have entreated him along
With us to watch the minutes of this night,
That if again this apparition come
He may approve our eyes and speak to it.
""",
"""
How now, Horatio? You tremble and look pale.
Is not this something more than fantasy?
What think you on ’t?
""",
"""
But soft, behold! Lo, where it comes again.
I’ll cross it though it blast me.—Stay, illusion!
"""
]

def run_models(sentence):
     # unigram frequency model
    unigram = ' '.join(map(shakespeare_synonym, sentence.split()))
    print('Unigram frequency model: {}'.format(unigram))

    # unigram model + bigram sentence fluency
    fluency = translate_to_shakespeare(sentence)
    print('Unigram model + bigram sentence fluency: {}'.format(fluency))

# create --shell option

# 1. output of those 3 sentences
# 2. screenshot of interactive shell
# 3. push to github
# 4. create paragraph from baseline

from cmd import Cmd

class MyPrompt(Cmd):
    prompt = '> '
    intro = "Welcome! Enter a sentence to translate into Shakespeare"

    def default(self, sentence):
        run_models(sentence)

if __name__ == '__main__':
    for s, o in zip(SENTENCES, ORIGINALS):
        print('Original Shakespeare:')
        print(o)
        print('Translating No Fear translation back to Shakespeare:')
        print(s)
        run_models(s)
    # MyPrompt().cmdloop()
    # sentence = """
    # Horatio says we’re imagining it,
    # and won’t let himself believe anything about
    # this horrible thing that we’ve seen twice now.
    # That’s why I’ve begged him
    # to come on our shift tonight,
    # so that if the ghost appears
    # he can see what we see and speak to it.
    # """
    # sentence = """
    # That’s why I’ve begged him to come on our shift tonight
    # """
    # sentence = 'today is a good day'




