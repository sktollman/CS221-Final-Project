import gensim
import os
import re

DIR_NAME = 'Training Data'

# removes punctuation, converts to lowercase, and splits into words
def vectorize_sentence(s):
    return re.sub(r'[^\w\s]','',s).lower().split()

def create_model():
    sentences = []
    for f_name in os.listdir('Training Data'):
        if not f_name.endswith('.txt'): continue
        with open(DIR_NAME + '/' + f_name, 'r', encoding='utf-8') as f:
            sentences.extend(list(map(vectorize_sentence, f.read().split('\n'))))

    model = gensim.models.Word2Vec(sentences)
    model.save('synonyms.pkl')

    return model

try:
    model = gensim.models.Word2Vec.load('synonyms.pkl')
except:
    model = create_model()

def shakes_synonym(word):
    try:
        res = list(map(lambda x: x[0], model.most_similar(word)))
    except:
        res = [word] # if no matches
    return res

def psyn(word):
    print('{}: {}'.format(word, shakes_synonym(word)))

# psyn('love')
# psyn('hate')
# psyn('day')
# psyn('this')
# psyn('along')
