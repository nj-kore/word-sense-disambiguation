import numpy as np
from collections import defaultdict
from classifier import NNClassifier
from sklearn.preprocessing import LabelEncoder

USE_EMBEDDING = True
EMBED_DIM = 200


def get_data(path):
    file_train = open(path, 'r', encoding='utf-8')
    file_lines = file_train.readlines()
    X = []
    Y = []
    for i, line in enumerate(file_lines):
        line = line.lower()
        tokens = line.split()
        target = tokens[0]
        lemma = tokens[1].split('.')[0]
        #if lemma != 'keep' and lemma != 'physical':
        #    continue
        input_vector = tokens[1:]
        X.append(input_vector)
        Y.append(target)

    return X, Y




def preprocess_X(vocab, word_embeddings, X_data, lemma_enc):

    start_of_text = 2
    rel_words = 5
    X = np.empty((len(X_data), (EMBED_DIM * 2) + 1), dtype=float)
    for i, tokens in enumerate(X_data):
        lemma = tokens[0].split('.')[0]
        target_loc = int(tokens[1]) + start_of_text
        start_fwd_pos = max(start_of_text, target_loc - rel_words)
        start_bckwd_pos = min(target_loc + rel_words + 1, len(tokens) - 1)

        fwd_embedding = np.zeros(EMBED_DIM, dtype=float)
        for token in tokens[start_fwd_pos:target_loc]:
            fwd_embedding += word_embeddings[vocab[token]]

        bckwd_embedding = np.zeros(EMBED_DIM, dtype=float)
        for token in reversed(tokens[target_loc + 1:start_bckwd_pos]):
            bckwd_embedding += word_embeddings[vocab[token]]

        input_vector = np.zeros(EMBED_DIM * 2 + 1)
        input_vector[:EMBED_DIM] = fwd_embedding
        input_vector[EMBED_DIM:-1] = bckwd_embedding

        input_vector = input_vector.tolist()
        input_vector[-1] = lemma_enc[lemma]

        X[i] = input_vector
    return X

if __name__ == '__main__':
    file_glove = open('data/glove.6B/glove.6B.200d.txt', 'r', encoding='utf-8')
    lines = file_glove.readlines()
    num_lines = len(lines)
    word_embeddings = np.zeros((num_lines + 1, EMBED_DIM), dtype=float)
    vocab = defaultdict(lambda: num_lines)
    vocab_rev = defaultdict(lambda: '<UNKNOWN>')

    if USE_EMBEDDING:
        for i, line in enumerate(lines):
            embed_split = line.split()
            word = embed_split[0]
            vocab[word] = i
            vocab_rev[i] = word

            embed_vals = np.asarray(embed_split[1:], dtype=float)

            word_embeddings[i, :] = embed_vals

        word_embeddings[num_lines, :] = np.zeros(EMBED_DIM)

    file_glove.close()

    x_train, y_train = get_data('data/wsd_train.txt')
    x_test, _ = get_data('data/wsd_test_blind.txt')

    lemma_enc = dict()
    lemma_idx = 0
    for i, y in enumerate(set(y_train)):
        lemma = y.split('%')[0]
        if lemma not in lemma_enc:
            lemma_enc[lemma] = lemma_idx
            lemma_idx += 1

    x_train = preprocess_X(vocab, word_embeddings, x_train, lemma_enc)
    x_test = preprocess_X(vocab, word_embeddings, x_test, lemma_enc)


    classifier = NNClassifier()
    lblenc = LabelEncoder()
    lblenc.fit(y_train)
    y_train_enc = lblenc.transform(y_train)
    classifier.fit(x_train, y_train_enc)


"""
def e(, vocab, word_embeddings):
    line = line.lower()
    tokens = line.split()
    target = tokens[0]
    lemma = tokens[1].split('.')[0]
    rel_words = 3
    start_of_text = 3
    target_loc = int(tokens[2]) + start_of_text
    start_fwd_pos = max(start_of_text, target_loc - rel_words)
    start_bckwd_pos = min(target_loc + rel_words + 1, len(tokens) - 1)

    fwd_embedding = np.zeros(EMBED_DIM, dtype=float)
    for token in tokens[start_fwd_pos:target_loc]:
        fwd_embedding += word_embeddings[vocab[token]]

    bckwd_embedding = np.zeros(EMBED_DIM, dtype=float)
    for token in reversed(tokens[target_loc + 1:start_bckwd_pos]):
        bckwd_embedding += word_embeddings[vocab[token]]

    input_vector = np.zeros(EMBED_DIM * 2)
    input_vector[:EMBED_DIM] = fwd_embedding
    input_vector[EMBED_DIM:] = bckwd_embedding

    input_vector = input_vector.tolist()
    input_vector[-1] = vocab[lemma.split('.')[0]]
    return input_vector, target

"""