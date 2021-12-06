import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split

# Definitions
EMBED_DIM = 300
GLOVE_FILE = 'glove.6B/glove.6B.' + str(EMBED_DIM) + 'd.txt'
REL_WORDS = 4

class ClassifierParameters:
    """This method has been copied from Week1 self study notebook, with a few modifications"""

    """Container class to store the hyperparameters that control the training process."""
    # Proportion of data set aside for validation.
    val_size = 0.2
    # Computation device: 'cuda' or 'cpu'
    device = 'cpu'
    # Number of hidden units in the neural network.
    n_hidden_units = 300
    # Size of batches: how many documents to process in parallel.
    batch_size = 512
    # Learning rate in the optimizer.
    learning_rate = 5e-3
    # Weight decay (L2 regularization) in the optimizer (if necessary).
    decay = 1e-5
    # Dropout probability (if necessary).
    dropout = 0.5

def make_model(clf):
    """This method has been copied from Week1 self study notebook, with a few modifications"""

    """Method for creating the neural network model"""
    input_size = clf.input_size
    output_size = clf.n_classes
    hidden_size = clf.params.n_hidden_units
    model = nn.Sequential(
        nn.BatchNorm1d(input_size),
        nn.Dropout(p=clf.params.dropout),
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.BatchNorm1d(hidden_size),
        nn.Dropout(p=clf.params.dropout),
        nn.Linear(hidden_size, output_size),
        nn.Softmax(dim=1)
        )
    return model


class NNClassifier:
    """A classifier based on a neural network."""

    def __init__(self, n_epochs):
        self.params = ClassifierParameters()
        self.model_factory = make_model
        self.lowest_loss = torch.inf
        self.early_stopping_itr = 0
        self.n_epochs = n_epochs

    def preprocess(self, X, Y):
        """This method has been copied from Week1 self study notebook"""

        """Carry out the document preprocessing, then build `DataLoader`s for the
           training and validation sets."""

        # Split X and Y into training and validation sets.
        Xtrain, Xval, Ytrain, Yval = train_test_split(X, Y, test_size=self.params.val_size, random_state=0)

        train_dataset = list(zip(Xtrain, Ytrain))
        val_dataset = list(zip(Xval, Yval))

        # Now, create the data loaders. The user parameters specify the batch size.
        self.train_loader = DataLoader(train_dataset, batch_size=self.params.batch_size)
        self.val_loader = DataLoader(val_dataset, batch_size=self.params.batch_size)

    def fit(self, X, Y):

        """This method has been copied from Week1 self study notebook, with a few modifications"""

        """Train the model. We assume that a dataset and a model have already been provided."""
        par = self.params

        self.preprocess(X, Y)

        self.input_size = X.shape[1]
        self.n_classes = len(set(Y))

        # Create a new model using the previously provided factory.
        # The assumption is that this is a PyTorch neural network that
        # takes inputs and makes outputs of the right sizes.
        self.model = self.model_factory(self)

        # If we're using a GPU, put the model there.
        self.model.to(par.device)

        # Declare a loss function, in this case the cross-entropy.
        self.loss_func = nn.CrossEntropyLoss()

        # An optimizer for updating the neural network. We use the Adam optimizer.
        optimizer = torch.optim.Adam(self.model.parameters(), lr=par.learning_rate, weight_decay=par.decay)

        # We'll log the loss and accuracy scores encountered during training.

        # Use the tqdm library to get a progress bar. The progress bar object can be used like
        # any generator-like object.
        progress = tqdm(range(self.n_epochs), 'Epochs')

        # Go through the dataset for a given number of epochs.
        for epoch in progress:
            # Set the model in training mode. This affects some components that
            # behave differently at training and evaluation time, such as dropout
            # and various types of normalization (e.g. batch normalization). It is good
            # practice to include this even if you don't use any dropout or normalization.
            self.model.train()

            # Run the model on the training data. The model will be updated after each batch.
            # See below for the implementation of `epoch`.
            train_loss, train_acc = self.epoch(self.train_loader, optimizer)

            # Set the model in evaluation mode, again affecting dropout and normalization modules.
            self.model.eval()

            # Run the model on the validation data. For somewhat improved efficiency, we disable
            # gradient computation now since we are not going to update the model.
            with torch.no_grad():
                val_loss, val_acc = self.epoch(self.val_loader)



            # Show validation-set metrics on the progress bar.
            progress.set_postfix({'val_loss': f'{val_loss:.2f}', 'val_acc': f'{val_acc:.2f}'})

            # I use early stopping. If the classifier has not beaten its best validation score for 15 epochs,
            # the training is terminated early.
            if val_loss < self.lowest_loss:
                self.lowest_loss = val_loss
                self.early_stopping_itr = 0
            else:
                self.early_stopping_itr += 1

            if self.early_stopping_itr > 15:
                break

    def epoch(self, batches, optimizer=None):
        """This method has been copied from Week1 self study notebook, with a few modifications"""

        """Runs the neural network for one epoch, using the given batches.
        If an optimizer is provided, this is training data and we will update the model
        after each batch. Otherwise, this is assumed to be validation data.

        Returns the loss and accuracy over the epoch."""
        n_correct = 0
        n_instances = 0
        total_loss = 0

        # We iterate through the batches (typically from a data loader).
        # This will give us X, Y pairs, containing the input and output parts
        # of this batch, respectively.
        for Xbatch, Ybatch in batches:

            Xbatch = Xbatch.to(self.params.device)
            Ybatch = Ybatch.to(self.params.device)

            # Compute the predictions for this batch.
            scores = self.model(Xbatch.float())

            # Compute the loss for this batch.
            loss = self.loss_func(scores, Ybatch.long())

            total_loss += loss.item()

            # Compute the number of correct predictions, for the accuracy.
            guesses = scores.argmax(dim=1)
            n_correct += (guesses == Ybatch).sum().item()
            n_instances += Ybatch.shape[0]

            # If this is training data, update the model.
            if optimizer:
                # Reset the gradients.
                optimizer.zero_grad()
                # Run the backprop algorithm to compute the new gradients.
                loss.backward()
                # Update the model based on the gradients.
                optimizer.step()

        return total_loss / len(batches), n_correct / n_instances

    def predict(self, X):
        """This method has been copied from Week1 self study notebook, with a few modifications"""

        """Run a trained classifier on a set of instances and return the predictions."""

        # Build a DataLoader to generate the batches.
        loader = DataLoader(X, batch_size=self.params.batch_size)

        # Apply the model to all the batches and aggregate the predictions.
        self.model.eval()
        outputs = []
        with torch.no_grad():
            for Xbatch in loader:
                # Move the batch onto the GPU if we are using one.
                Xbatch = Xbatch.to(self.params.device)

                # Compute the output scores.
                scores = self.model(Xbatch.float())
                # scores should have the shape (batch_size, n_classes).

                # For each row, find the position of the highest score. This represents
                # the model's guess for this instance.
                # The output will have the shape (batch_size).
                guesses = scores.argmax(dim=1)

                # Move the result back onto the CPU and convert into a NumPy array,
                # and keep the result for later.
                outputs.append(guesses.cpu().numpy())

            # Finally, concatenate all output arrays.
            return np.hstack(outputs)


class MultiClassifier:

    """A classifier that is actually just an orchestrator for a set of sub classifiers. One sub classifier handles one
    target word lemma."""

    def __init__(self, n_epochs):
        self.classifiers = dict()
        self.main_lbl_encoder = LabelEncoder()
        self.label_encoders = dict()
        self.n_epochs = n_epochs

    def preprocess_X(self, X_data):
        """Preprocess X by summing embeddings around our target word"""

        # Index of where the text actually starts in the input data
        start_of_text = 2

        X = []
        for i, tokens in enumerate(X_data):

            # Location of target word
            target_loc = int(tokens[1]) + start_of_text

            # First word in our context window. Using Max to avoid going out of bounds
            start_fwd_pos = max(start_of_text, target_loc - REL_WORDS)

            # Last word in our context window. Using min to avoid going out of bounds
            start_bckwd_pos = min(target_loc + REL_WORDS + 1, len(tokens) - 1)

            # Create a vector that is going to be the sum of the relevant words before our target word
            fwd_embedding = np.zeros(EMBED_DIM, dtype=float)
            for token in tokens[start_fwd_pos:target_loc]:
                fwd_embedding += self.word_embeddings[self.vocab[token]]

            # Create a vector that is going to be the sum of the relevant words after our target word
            bckwd_embedding = np.zeros(EMBED_DIM, dtype=float)
            for token in reversed(tokens[target_loc + 1:start_bckwd_pos]):
                bckwd_embedding += self.word_embeddings[self.vocab[token]]

            # Create the input vector data point by concatenating the forward and backward embeddings
            input_vector = np.zeros(EMBED_DIM * 2)
            input_vector[:EMBED_DIM] = fwd_embedding
            input_vector[EMBED_DIM:] = bckwd_embedding

            input_vector = input_vector.tolist()

            X.append(input_vector)
        return X

    def init_embeddings(self):

        """Extract the embeddings and create a vocab that maps to the correct embedding"""

        print("extracting embeddings...")

        file_glove = open(GLOVE_FILE, 'r', encoding='utf-8')
        lines = file_glove.readlines()
        num_lines = len(lines)

        # I create one more row for the embeddings that is dedicated to a word not present in the vocab
        self.word_embeddings = np.zeros((num_lines + 1, EMBED_DIM), dtype=float)
        self.vocab = defaultdict(lambda: num_lines)

        #  Read all word -> embedding mappings and enter into our data structures
        for i, line in enumerate(lines):
            embed_split = line.split()
            word = embed_split[0]
            self.vocab[word] = i
            embed_vals = np.asarray(embed_split[1:], dtype=float)
            self.word_embeddings[i, :] = embed_vals

        # Define a non existing word to map to an array of zeros
        self.word_embeddings[num_lines, :] = np.zeros(EMBED_DIM)
        file_glove.close()


    def fit(self, X, Y):

        self.init_embeddings()

        # Here I go through Y and register in categories which lemma each datapoint in Y corresponds to. This is so
        # that the correct datapoints will be send to the correct classifier, specific for that word.
        # At the same time, I'm constructing a lemma_enc dict which maps a lemma to an integer
        categories = []
        self.lemma_enc = dict()
        self.num_categories = 0
        for i, y in enumerate(Y):
            lemma = y.split('%')[0]
            if lemma not in self.lemma_enc:
                self.lemma_enc[lemma] = self.num_categories
                categories.append(self.num_categories)
                self.num_categories += 1
            else:
                categories.append(self.lemma_enc[lemma])

        print('preprocessing training data...')
        X = self.preprocess_X(X)

        # Fit LabelEncoder
        self.main_lbl_encoder.fit(Y)
        Y = self.main_lbl_encoder.transform(Y)

        # Seperate the data into different arrays so that one array contains datapoints relating to one lemma
        X_sep = [[] for _ in range(self.num_categories)]
        Y_sep = [[] for _ in range(self.num_categories)]

        # Append to correct array by looking at what category (lemma) the datapoint belongs to
        for i in range(len(X)):
            X_sep[categories[i]].append(X[i])
            Y_sep[categories[i]].append(Y[i])

        # Go through all the categories (lemmas) and create and train one classifier for each. We must also know
        # the label encoder for this classifier so we can map the results back
        for i in range(self.num_categories):
            classifier = NNClassifier(self.n_epochs)
            lblenc = LabelEncoder()
            lblenc.fit(Y_sep[i])
            y_sep_i_enc = lblenc.transform(Y_sep[i])
            classifier.fit(np.array(X_sep[i]), y_sep_i_enc)
            self.classifiers[i] = classifier
            self.label_encoders[i] = lblenc



    def predict(self, X):

        # Arrays for storing which index of X belongs to which category.
        X_categories = [[] for _ in range(self.num_categories)]

        # Register what indexes of X should be in which array
        for i in range(len(X)):
            lemma = X[i][0].split('.')[0]
            X_categories[self.lemma_enc[lemma]].append(i)

        print('preprocessing test data...')
        X = self.preprocess_X(X)

        # Array for storing the different partitions of X depending on what category they belong to
        X_sep = [[] for _ in range(self.num_categories)]

        # Sort X into partitions
        for i in range(self.num_categories):
            for j in X_categories[i]:
                X_sep[i].append(X[j])


        # Go through and make predictions for each category.
        all_predictions = ["" for _ in range(len(X))]
        for i in range(self.num_categories):
            predictions = self.classifiers[i].predict(np.array(X_sep[i]))

            # Translate to MultiClassifiers encoding system
            predictions = self.label_encoders[i].inverse_transform(predictions)

            # Translate to normal words
            predictions = self.main_lbl_encoder.inverse_transform(predictions)
            for j, p in enumerate(predictions):
                all_predictions[X_categories[i][j]] = p

        return all_predictions


