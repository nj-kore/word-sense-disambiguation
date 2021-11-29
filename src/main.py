from classifier_1 import MultiClassifier
from classifier_2 import RNNClassifier
import argparse
import sys

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", help="select classifier [1 or 2]", type=int)
    args = parser.parse_args()


    #classifier = NNClassifier()
    x_train, y_train = get_data('data/wsd_train.txt')
    x_test, _ = get_data('data/wsd_test_blind.txt')

    classifier = None
    if args.classifier == 1:
        classifier = MultiClassifier()
    elif args.classifier == 2:
        classifier = RNNClassifier()
    else:
        print("No such classifier found")
        exit(-1)

    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)

    f = open('predictions_1.txt', 'w', encoding='utf-8')
    for p in predictions:
        f.write(p + '\n')
