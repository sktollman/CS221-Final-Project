# StackOverflow question: https://stackoverflow.com/questions/51123481/how-to-build-a-language-model-using-lstm-that-assigns-probability-of-occurence-f
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential, load_model
import numpy as np
import re

def prepare_sentence(seq, maxlen):
    # Pads seq and slides windows
    x = []
    y = []
    for i, w in enumerate(seq):
        x_padded = pad_sequences([seq[:i]],
                                 maxlen=maxlen - 1,
                                 # Pads before each sequence
                                 padding='pre')[0]
        x.append(x_padded)
        y.append(w)
    return x, y

# removes punctuation and converts to lowercase
def normalize_sentence(s):
    return re.sub(r'[^\w\s]','',s).lower()

with open('alllines.txt', 'r', encoding='utf-8') as f:
    data = list(map(normalize_sentence, f.read().split('\n')))

# Preprocess data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)
vocab = tokenizer.word_index
seqs = tokenizer.texts_to_sequences(data)

# Slide windows over each sentence
maxlen = max([len(seq) for seq in seqs])

def create_model():
    x = []
    y = []
    for seq in seqs:
        x_windows, y_windows = prepare_sentence(seq, maxlen)
        x += x_windows
        y += y_windows
    x = np.array(x)
    y = np.array(y) - 1
    y = np.eye(len(vocab))[y]  # One hot encoding

    # Define model
    model = Sequential()
    model.add(Embedding(input_dim=len(vocab) + 1, # vocabulary size. Adding an
                                                  # extra element for <PAD> word
                        output_dim=5,  # size of embeddings
                        input_length=maxlen - 1)) # length of the padded seqs
    model.add(LSTM(10))
    model.add(Dense(len(vocab), activation='softmax'))
    model.compile('rmsprop', 'categorical_crossentropy')

    # Train network
    model.fit(x, y, epochs=1000)
    model.save('model.h5')

    return model

# Only generate model once because it is expensive
try:
    model = load_model('model.h5')
except:
    model = create_model()

# Compute probability of occurence of a sentence
def score_sentence(sentence="So shaken as we are, so wan with care,",
    verbose=False):
    sentence = normalize_sentence(sentence)
    tok = tokenizer.texts_to_sequences([sentence])[0]
    x_test, y_test = prepare_sentence(tok, maxlen)
    x_test = np.array(x_test)
    y_test = np.array(y_test) - 1  # The word <PAD> does not have a class
    p_pred = model.predict(x_test)
    vocab_inv = {v: k for k, v in vocab.items()}
    log_p_sentence = 0
    for i, prob in enumerate(p_pred):
        word = vocab_inv[y_test[i]+1]  # Index 0 from vocab is reserved to <PAD>
        history = ' '.join([vocab_inv[w] for w in x_test[i, :] if w != 0])
        prob_word = prob[y_test[i]]
        log_p_sentence += np.log(prob_word)
        if verbose: print('P(w={}|h={})={}'.format(word, history, prob_word))
    result = np.exp(log_p_sentence)
    if verbose: print('Prob. sentence: {}'.format(result))

    return result

score_sentence(verbose=True)
