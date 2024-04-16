
#
# X, y = [], []
# load data set
#     for each folder in "asthma-dataset-1" folder
#     (or from augmented folder if it has data )
#         X = load csv
#         each folder is one class
#         y = folder name
#         convert y to one hot encoding
#         the csv library should parse them into floats
#             if not, do it manually
#
# shuffle the data
# split into training/validation randomly
#
# (unsure if we have enough CSVs for a test set as well)
#
# if we want to batch our data, we can do that with DataLoader
#     (from torch.utils.data import DataLoader...)
# but I dont think this is necessary
#      its fine to do batch size of 1 at this stage


# This code is from my lab07. adapt it to our scenario
# we do not have embeddings or vocabulary...
# we just have num_columns and num_classes
#   which we can get from the loaded data
# sp

import torch.nn as nn

class MyRNN(nn.Module):
    def __init__(self, num_cols, hidden_size, num_classes):
        super(MyRNN, self).__init__()
        self.num_cols = num_cols
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.lstm = nn.LSTM(num_cols, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, X):
        out, h = self.rnn(X)
        av = torch.mean(out, dim=1)
        return self.fc(av)



# copied from lab07 - we should cite where we use existing code
# unsure if this has to be modified to ours
def accuracy(model, dataset, max=1000):
    """
    Estimate the accuracy of `model` over the `dataset`.
    We will take the **most probable class**
    as the class predicted by the model.

    Parameters:
        `model`   - An object of class nn.Module
        `dataset` - A dataset of the same type as `train_data`.
        `max`     - The max number of samples to use to estimate
                    model accuracy

    Returns: a floating-point value between 0 and 1.
    """
    model.eval()

    correct, total = 0, 0
    dataloader = DataLoader(dataset,
                            batch_size=1,  # use batch size 1 to prevent padding
                            collate_fn=collate_batch)
    for i, (x, t) in enumerate(dataloader):
        x, t = x.to(device), t.to(device)  # Move data to the correct device
        z = model(x)
        y = torch.argmax(z, axis=1)
        correct += int(torch.sum(t == y))
        total   += 1
        if i >= max:
            break

    model.train()

    return correct / total

import torch.optim as optim
import matplotlib.pyplot as plt


# again copied from lab07
def train_model(model,                # an instance of MLPModel
                train_data,           # training data
                val_data,             # validation data
                learning_rate=0.001,
                batch_size=100,
                num_epochs=10,
                plot_every=50,        # how often (in # iterations) to track metrics
                plot=True,            # whether to plot the training curve
                clip_grad_norm=False):
    print("training with clip_grad_norm=", clip_grad_norm)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               collate_fn=collate_batch,
                                               shuffle=True) # reshuffle minibatches every epoch
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)#, weight_decay=1e-6)
    model.to(device)

    # these lists will be used to track the training progress
    # and to plot the training curve
    iters, train_loss, train_acc, val_acc = [], [], [], []
    iter_count = 0 # count the number of iterations that has passed

    try:
        for e in range(num_epochs):
            for i, (texts, labels) in enumerate(train_loader):
                texts = texts.to(device)
                labels = labels.to(device)

                z = model(texts)

                loss = criterion(z, labels)

                loss.backward() # propagate the gradients
                if clip_grad_norm:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step() # update the parameters
                optimizer.zero_grad() # clean up accumualted gradients

                iter_count += 1
                if iter_count % plot_every == 0:
                    iters.append(iter_count)
                    ta = accuracy(model, train_data)
                    va = accuracy(model, val_data)
                    train_loss.append(float(loss))
                    train_acc.append(ta)
                    val_acc.append(va)
                    print(f"{iter_count} Loss: {loss:.3f} Train Acc: {ta:.3f} Val Acc: {va:.3f}")
    finally:
        # This try/finally block is to display the training curve
        # even if training is interrupted
        if plot:
            plt.figure()
            plt.plot(iters[:len(train_loss)], train_loss)
            plt.title("Loss over iterations")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")

            plt.figure()
            plt.plot(iters[:len(train_acc)], train_acc)
            plt.plot(iters[:len(val_acc)], val_acc)
            plt.title("Accuracy over iterations")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.legend(["Train", "Validation"])

model = MyRNN(vocab_size=len(vocab),
              emb_size=300,
              hidden_size=64,
              num_classes=2)

train_model(model, train_data_indices[0:10], train_data_indices[0:10], batch_size=1, num_epochs=10, plot_every=10, plot=True, clip_grad_norm=True)
