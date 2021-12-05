from classifier_1 import MultiClassifier
from classifier_2 import RNNClassifier
import matplotlib.pyplot as plt
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

    clf = None
    if args.classifier == 1:
        clf = MultiClassifier()
    elif args.classifier == 2:
        clf = RNNClassifier()
    else:
        print("No such classifier found")
        exit(-1)

    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)

    f = open('pred', 'w', encoding='utf-8')
    for p in predictions:
        f.write(p + '\n')

    plt.style.use('seaborn')

    x = range(len(clf.history['train_loss']))
    fig, ax = plt.subplots(1, 2, figsize=(2 * 6, 1 * 6))
    ax[0].plot(x, clf.history['train_loss'], x, clf.history['val_loss'])
    ax[0].legend(['train loss', 'val loss'])
    ax[1].plot(x, clf.history['train_acc'], x, clf.history['val_acc'])
    ax[1].legend(['train acc', 'val acc'])
    plt.savefig("classifier_" + str(args.classifier) + "_results.png")
