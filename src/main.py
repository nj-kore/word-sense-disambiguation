from classifier_1 import MultiClassifier
from classifier_2 import RNNClassifier
import matplotlib.pyplot as plt
import argparse

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
        input_vector = tokens[1:]
        X.append(input_vector)
        Y.append(target)

    return X, Y

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", help="select classifier [1 or 2]", type=int)
    parser.add_argument("--n_epochs", help="number of epochs", type=int)
    args = parser.parse_args()

    print('extracting data into training and test data...')
    x_train, y_train = get_data('data/wsd_train.txt')
    x_test, _ = get_data('data/wsd_test_blind.txt')

    clf = None
    if args.classifier == 1:
        clf = MultiClassifier(args.n_epochs)
    elif args.classifier == 2:
        clf = RNNClassifier(args.n_epochs)
    else:
        print("No such classifier found")
        exit(-1)

    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)

    f = open("classifier_" + str(args.classifier) + "_results.txt", 'w', encoding='utf-8')
    for p in predictions:
        f.write(p + '\n')

    if (args.classifier == 2):
        plt.style.use('seaborn')
        x = range(len(clf.history['train_loss']))
        fig, ax = plt.subplots(1, 2, figsize=(2 * 6, 1 * 6))
        ax[0].plot(x, clf.history['train_loss'], x, clf.history['val_loss'])
        ax[0].legend(['train loss', 'val loss'])
        ax[1].plot(x, clf.history['train_acc'], x, clf.history['val_acc'])
        ax[1].legend(['train acc', 'val acc'])
        plt.savefig("classifier_" + str(args.classifier) + "_plot.png")
