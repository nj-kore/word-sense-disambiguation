import numpy as np
from collections import defaultdict
from classifier import NNClassifier, MultiClassifier
from sklearn.preprocessing import LabelEncoder

USE_EMBEDDING = True


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

if __name__ == '__main__':

    #classifier = NNClassifier()
    x_train, y_train = get_data('data/wsd_train.txt')

    multi_classifier = MultiClassifier()
    multi_classifier.fit(x_train, y_train)

    x_test, _ = get_data('data/wsd_test_blind.txt')

    predictions = multi_classifier.predict(x_test)

    f = open('predictions_1.txt', 'w', encoding='utf-8')
    for p in predictions:
        f.write(p + '\n')


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