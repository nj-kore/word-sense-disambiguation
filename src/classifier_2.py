import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split

# definitions
EMBED_DIM = 100
GLOVE_FILE = 'glove.6B/glove.6B.' + str(EMBED_DIM) + 'd.txt'
REL_WORDS = 10

class LSTMModel(nn.Module):

    """Model for an RNN. """
    def __init__(self, clf):
        super(LSTMModel, self).__init__()

        self.rnn_input_size = clf.input_size
        self.hidden_units = clf.params.hidden_units
        self.lstm_hidden_layers = clf.params.lstm_hidden_layers
        self.input_bn = nn.BatchNorm1d(REL_WORDS+1)

        # The BI-LSTM has been split up to two networks, as described in my report
        self.lstm_fwd = nn.LSTM(self.rnn_input_size, self.hidden_units,
                                self.lstm_hidden_layers, batch_first=True)

        self.lstm_bckwd = nn.LSTM(self.rnn_input_size, self.hidden_units,
                                  self.lstm_hidden_layers, batch_first=True)

        # Output fully connected network
        # Having the "+1" seemed to yield slighlty better results. Perhaps keeping the layers equal size makes for
        # a better result. This theory might be wrong however
        self.seq = nn.Sequential(
            nn.BatchNorm1d(2*self.hidden_units+1),
            nn.Linear(2*self.hidden_units+1, 2*self.hidden_units+1),
            nn.ReLU(),
            nn.BatchNorm1d(2*self.hidden_units+1),
            nn.Linear(2*self.hidden_units+1, clf.n_classes),
            nn.Softmax(dim=1)
        )


    def forward(self, x):

        # Feed into batch norm
        x_fwd = self.input_bn(x[:, 0])
        x_bckwd = self.input_bn(x[:, 1])

        # Extract the lemma (encoded)
        lemma = x[:, 2, 0, 0]

        # Init hidden state and cell state to zeros
        h0 = torch.zeros(self.lstm_hidden_layers, x.size(0), self.hidden_units, requires_grad=True)
        c0 = torch.zeros(self.lstm_hidden_layers, x.size(0), self.hidden_units, requires_grad=True)

        # Feed into the BI-LSTM
        _, (hn_fwd, cn_fwd) = self.lstm_fwd(x_fwd, (h0.detach(), c0.detach()))
        _, (hn_bckwd, cn_bckwd) = self.lstm_bckwd(x_bckwd, (h0.detach(), c0.detach()))

        # Create a new input to the output network by concatenating hidden states and adding the encoded lemma at the
        # end
        inp = torch.zeros((x.size(0), 2*self.hidden_units + 1))
        hidden_concat = torch.cat((hn_fwd[0], hn_bckwd[0]), dim=1)
        inp[:, :-1] = hidden_concat
        inp[:, -1] = lemma

        # Feed into output network
        out = self.seq(inp)
        return out


class ClassifierParameters:
    """This method has been copied from Week1 self study notebook, with a few modifications"""

    """Container class to store the hyperparameters that control the training process."""
    # Proportion of data set aside for validation.
    val_size = 0.2
    # Computation device: 'cuda' or 'cpu'
    device = 'cpu'
    # Number of hidden units in the neural network.
    hidden_units = 150
    # Number of hidden layers for the rnn network
    rnn_hidden_layers = 1
    # Size of batches: how many documents to process in parallel.
    batch_size = 512
    # Learning rate in the optimizer.
    learning_rate = 5e-3
    # Weight decay (L2 regularization) in the optimizer (if necessary).
    decay = 1e-5
    # Dropout probability (if necessary).
    dropout = 0.5


# Wrapper for creating the network
def make_model(clf):
    return LSTMModel(clf)


class LSTMClassifier:
    """A classifier based on a Long Short Term Memory network."""

    def __init__(self, n_epochs):
        self.params = ClassifierParameters()
        self.model_factory = make_model
        self.main_lbl_encoder = LabelEncoder()
        self.n_epochs = n_epochs


    def init_embeddings(self):

        """Extract the embeddings and create a vocab that maps to the correct embedding"""

        print("extracting embeddings...")

        file_glove = open(GLOVE_FILE, 'r', encoding='utf-8')
        lines = file_glove.readlines()
        num_lines = len(lines)

        # I create one more row for the embeddings that is dedicated to a word not present in the vocab
        self.word_embeddings = np.zeros((num_lines + 1, EMBED_DIM), dtype=float)
        self.vocab = defaultdict(lambda: num_lines)

        # Read all word -> embedding mappings and enter into our data structures
        for i, line in enumerate(lines):
            embed_split = line.split()
            word = embed_split[0]
            self.vocab[word] = i
            embed_vals = np.asarray(embed_split[1:], dtype=float)
            self.word_embeddings[i, :] = embed_vals

        # Define a non existing word to map to an array of zeros
        self.word_embeddings[num_lines, :] = np.zeros(EMBED_DIM)
        file_glove.close()

    def _preprocess_X(self, X_data):
        """Preprocess X by saving embeddings around our target word"""

        # Index of where the text actually starts in the input data
        start_of_text = 2
        X = []
        for i, tokens in enumerate(X_data):

            # Extract the lemma
            lemma = tokens[0].split('.')[0]

            # Location of target word
            target_loc = int(tokens[1]) + start_of_text

            # First word in our context window. Using Max to avoid going out of bounds
            start_fwd_pos = max(start_of_text, target_loc - REL_WORDS)

            # Last word in our context window. Using Minx to avoid going out of bounds
            start_bckwd_pos = min(target_loc + REL_WORDS + 1, len(tokens) - 1)
            fwd_embeddings = np.zeros((REL_WORDS+1, EMBED_DIM), dtype=float)

            # Offset is just how many words we are about to enter into our array. If we have gone out of bounds and the
            # number of relevant words are less than what we want, this offset makes sure that the words that we enter
            # at still put at the correct location. For instance, if we want to enter 5 words before our target word,
            # but there are only 2 words available, then the first 3 words will be filled with an array of zero.
            offset = target_loc - start_fwd_pos

            # Go through the words from start_fwd_pos to target_loc and enter the word embeddings as data. The target
            # word is included.
            for j, token in enumerate(tokens[start_fwd_pos:target_loc+1]):
                fwd_embeddings[j + REL_WORDS - offset] = self.word_embeddings[self.vocab[token]]

            # Same reasoning but for the words behind the target word (and including the target word)
            bckwd_embeddings = np.zeros((REL_WORDS+1, EMBED_DIM), dtype=float)
            offset = start_bckwd_pos - target_loc - 1
            for j, token in enumerate(reversed(tokens[target_loc:start_bckwd_pos])):
                bckwd_embeddings[j + REL_WORDS - offset] = self.word_embeddings[self.vocab[token]]

            # The lemma_arr has a dimension of (REL_WORDS+1, EMBED_DIM) but it uses one row with the word
            # embedding for the current lemma, and one position containing the encoded index for the lemma. The
            # lemma_arr had to be this size to avoid problems with tensors, but a lot of rows remained unused for
            # this array
            lemma_arr = np.zeros((REL_WORDS+1, EMBED_DIM), dtype=float)
            lemma_arr[0, 0] = self.lemma_enc[lemma]
            lemma_arr[1] = self.word_embeddings[self.vocab[lemma]]

            # Append the resulting data into X. This is one data point for one target word
            X.append([fwd_embeddings, bckwd_embeddings, lemma_arr])
        return np.array(X)

    def preprocess(self, X, Y):
        """Carry out the document preprocessing, then build `DataLoader`s for the
           training and validation sets."""

        # I create a dictionary that maps a lemma to an index. Used for preprocess_x(x)
        self.lemma_enc = dict()
        self.num_lemmas = 0
        for i, y in enumerate(Y):
            lemma = y.split('%')[0]
            if lemma not in self.lemma_enc:
                self.lemma_enc[lemma] = self.num_lemmas
                self.num_lemmas += 1

        X = self._preprocess_X(X)

        # Encoder so that we can map each target to a number instead
        self.main_lbl_encoder.fit(Y)

        # Convert Y to the encoded version
        Y = self.main_lbl_encoder.transform(Y)

        """The below code has been copied from Week1 self study notebook"""
        # Split X and Y into training and validation sets.
        Xtrain, Xval, Ytrain, Yval = train_test_split(X, Y, test_size=self.params.val_size, random_state=0)

        train_dataset = list(zip(Xtrain, Ytrain))
        val_dataset = list(zip(Xval, Yval))

        # Create loaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.params.batch_size)
        self.val_loader = DataLoader(val_dataset, batch_size=self.params.batch_size)

        return X, Y

    def fit(self, X, Y):

        """This method has been copied from Week1 self study notebook, with a few modifications"""

        """Train the model"""
        par = self.params
        self.init_embeddings()
        X, Y = self.preprocess(X, Y)

        self.input_size = X.shape[3]
        self.n_classes = len(set(Y))

        # Create our Neural Network model
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
        progress = tqdm(range(self.n_epochs), 'Epochs')

        # Go through the dataset for a given number of epochs.
        for epoch in progress:
            t0 = time.time()

            # Set the model in training mode.
            self.model.train()

            # Run the model on the training data. The model will be updated after each batch.
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
            progress.set_postfix({'val_loss': f'{val_loss:.2f}', 'val_acc': f'{val_acc:.2f}', 'train_acc': f'{train_acc:.2f}'})


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

            # If we're using the GPU, move the batch there.
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
        X = self._preprocess_X(X)

        # Build a DataLoader to generate the batches
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

                # For each row, find the position of the highest score. This represents
                # the model's guess for this instance.
                # The output will have the shape (batch_size).
                guesses = scores.argmax(dim=1)

                # Move the result back onto the CPU and convert into a NumPy array,
                # and keep the result for later.
                outputs.append(guesses.cpu().numpy())

            # Finally, concatenate all output arrays.
            outputs = np.hstack(outputs)
            return self.main_lbl_encoder.inverse_transform(outputs)

