import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split


class RNNModel(nn.Module):
    def __init__(self, clf):
        super(RNNModel, self).__init__()

        # Building your RNN
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, input_dim)
        # batch_dim = number of samples per batch

        self.rnn_input_size = clf.input_size
        self.rnn_hidden_units = clf.params.rnn_hidden_units
        self.rnn_hidden_layers = clf.params.rnn_hidden_layers
        self.fc_hidden_units = clf.params.fc_hidden_units

        self.rnn = nn.RNN(self.rnn_input_size, self.rnn_hidden_units,
                          self.rnn_hidden_layers, batch_first=True, nonlinearity='relu')

        self.output = nn.Sequential(
            nn.BatchNorm1d(self.rnn_input_size),
            nn.Linear(self.rnn_input_size, self.fc_hidden_units),
            nn.ReLU(),
            nn.BatchNorm1d(self.fc_hidden_units),
            nn.Dropout(p=clf.params.dropout),
            nn.Linear(self.fc_hidden_units, clf.n_classes),
            nn.Softmax(dim=1)
        )


    def forward(self, x):
        # Initialize hidden state with zeros
        # (layer_dim, batch_size, hidden_dim)
        h0 = torch.zeros(self.rnn_hidden_layers, x.size(0), self.rnn_hidden_units).requires_grad_()

        # We need to detach the hidden state to prevent exploding/vanishing gradients
        # This is part of truncated backpropagation through time (BPTT)
        out, hn = self.rnn(x, h0.detach())

        out = self.output(hn[:, -1, :])
        # Index hidden state of last time step
        # out.size() --> 100, 28, 10
        # out[:, -1, :] --> 100, 10 --> just want last time step hidden states!

        # out.size() --> 100, 10
        return out


class ClassifierParameters:
    """Container class to store the hyperparameters that control the training process."""
    # Proportion of data set aside for validation.
    val_size = 0.2
    # Computation device: 'cuda' or 'cpu'
    device = 'cpu'
    # Number of hidden units in the neural network.
    rnn_hidden_units = 50
    # Number of hidden layers for the rnn network
    rnn_hidden_layers = 1
    # Number of training epochs.
    fc_hidden_units = 50

    n_epochs = 100
    # Size of batches: how many documents to process in parallel.
    batch_size = 512
    # Learning rate in the optimizer.
    learning_rate = 5e-3
    # Weight decay (L2 regularization) in the optimizer (if necessary).
    decay = 1e-5
    # Dropout probability (if necessary).
    dropout = 0.5

def make_model(clf):
    return RNNModel(clf)

EMBED_DIM = 50
class RNNClassifier:
    """A classifier based on a neural network."""

    def __init__(self):
        self.params = ClassifierParameters()
        self.model_factory = make_model
        self.lowest_loss = torch.inf
        self.early_stopping_itr = 0
        self.main_lbl_encoder = LabelEncoder()


    def init_embeddings(self):
        file_glove = open('data/glove.6B/glove.6B.50d.txt', 'r', encoding='utf-8')
        lines = file_glove.readlines()
        num_lines = len(lines)
        self.word_embeddings = np.zeros((num_lines + 1, EMBED_DIM), dtype=float)
        self.vocab = defaultdict(lambda: num_lines)
        self.vocab_rev = defaultdict(lambda: '<UNKNOWN>')

        for i, line in enumerate(lines):
            embed_split = line.split()
            word = embed_split[0]
            self.vocab[word] = i
            self.vocab_rev[i] = word

            embed_vals = np.asarray(embed_split[1:], dtype=float)

            self.word_embeddings[i, :] = embed_vals

        self.word_embeddings[num_lines, :] = np.zeros(EMBED_DIM)
        file_glove.close()

    def _preprocess_X(self, X_data):
        start_of_text = 2
        rel_words = 5
        X = np.empty((len(X_data), rel_words, EMBED_DIM), dtype=float)
        #X = []
        for i, tokens in enumerate(X_data):
            lemma = tokens[0].split('.')[0]
            target_loc = int(tokens[1]) + start_of_text
            start_fwd_pos = max(start_of_text, target_loc - rel_words)
            start_bckwd_pos = min(target_loc + rel_words + 1, len(tokens) - 1)

            fwd_embeddings = np.zeros((rel_words, EMBED_DIM), dtype=float)
            offset = target_loc - start_fwd_pos
            for j, token in enumerate(tokens[start_fwd_pos:target_loc]):
                fwd_embeddings[j + rel_words - offset] = self.word_embeddings[self.vocab[token]]

            bckwd_embeddings = np.zeros((rel_words, EMBED_DIM), dtype=float)
            offset = start_bckwd_pos - target_loc - 1
            for j, token in enumerate(reversed(tokens[target_loc + 1:start_bckwd_pos])):
                bckwd_embeddings[j + rel_words - offset] = self.word_embeddings[self.vocab[token]]

            input_vector = fwd_embeddings#, bckwd_embeddings]

            X[i] = input_vector
        return X

    def preprocess(self, X, Y):
        """Carry out the document preprocessing, then build `DataLoader`s for the
           training and validation sets."""
        X = self._preprocess_X(X)
        self.main_lbl_encoder.fit(Y)
        Y = self.main_lbl_encoder.transform(Y)
        # Split X and Y into training and validation sets. We
        # apply the utility function we used previously.
        Xtrain, Xval, Ytrain, Yval = train_test_split(X, Y, test_size=self.params.val_size, random_state=0)

        # As discussed above, a dataset simply needs to behave like a list: it should
        # be aware of its length and be able to index:
        # dataset[position] should give an x,y pair (input and output).
        # This means that we can simply use lists here!
        train_dataset = list(zip(Xtrain, Ytrain))
        val_dataset = list(zip(Xval, Yval))

        # Now, create the data loaders. The user parameters specify the batch size.
        self.train_loader = DataLoader(train_dataset, batch_size=self.params.batch_size)
        self.val_loader = DataLoader(val_dataset, batch_size=self.params.batch_size)

        return X, Y

    def fit(self, X, Y):
        """Train the model. We assume that a dataset and a model have already been provided."""
        par = self.params
        self.init_embeddings()
        X, Y = self.preprocess(X, Y)

        self.input_size = X.shape[2]
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
        self.history = defaultdict(list)

        # Use the tqdm library to get a progress bar. The progress bar object can be used like
        # any generator-like object.
        progress = tqdm(range(par.n_epochs), 'Epochs')

        # Go through the dataset for a given number of epochs.
        for epoch in progress:
            t0 = time.time()

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


            t1 = time.time()

            # Store some evaluation metrics in the history object.
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['time'].append(t1 - t0)

            # Show validation-set metrics on the progress bar.
            progress.set_postfix({'val_loss': f'{val_loss:.2f}', 'val_acc': f'{val_acc:.2f}'})

            if val_loss < self.lowest_loss:
                self.lowest_loss = val_loss
                self.early_stopping_itr = 0
            else:
                self.early_stopping_itr += 1

            if self.early_stopping_itr > 10:
                break

    def epoch(self, batches, optimizer=None):
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

            # Xbatch is a tensor of shape (batch_size, input_size).
            # Ybatch has the shape (batch_size).

            # If we're using the GPU, move the batch there.
            Xbatch = Xbatch.to(self.params.device)
            Ybatch = Ybatch.to(self.params.device)
            yo = Xbatch.shape
            # Compute the predictions for this batch.
            scores = self.model(Xbatch.float())

            # If the previous step was implemented correctly, your scores
            # tensor should have the shape (batch_size, n_classes).

            # Compute the loss for this batch. Note: various loss functions
            # behave differently, depending on whether they aggregate or not
            # (that is, by summing or averaging).
            # In the end, the loss value needs to be a single number so
            # that we can compute gradients and update our model later.
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
        """Run a trained classifier on a set of instances and return the predictions."""

        # Build a DataLoader to generate the batches, as above except that now we don't have Y.
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