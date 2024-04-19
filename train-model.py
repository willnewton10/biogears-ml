import torch
from torch.utils.data import DataLoader
from load_data import load_data, split_dataset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import global_constants


dataset = load_data("asthma-dataset-1")

train_data, val_data, test_data = split_dataset(dataset)


train_dataloader = DataLoader(train_data, batch_size=10, shuffle=True)


class MyRNN(nn.Module):
    def __init__(self, num_cols, hidden_size, num_classes):
        super(MyRNN, self).__init__()
        self.num_cols = num_cols
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.lstm = nn.LSTM(num_cols, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, X):
        out, h = self.lstm(X.float())
        av = torch.mean(out, dim=1)
        return self.fc(av)



# copied from lab07
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
                            )
    for i, (x, t) in enumerate(dataloader):
        # x, t = x.to(device), t.to(device)  # Move data to the correct device
        z = model(x)
        y = torch.argmax(z, axis=1)
        correct += int(torch.sum(torch.argmax(t) == y))
        total   += 1
        if i >= max:
            break

    return correct / total


def train_model(model,                # an instance of MLPModel
                train_data,           # training data
                val_data,             # validation data
                learning_rate=0.01,
                weight_decay=-1,
                batch_size=10,
                num_epochs=5,
                plot_every=10,        # how often (in # iterations) to track metrics
                plot=True,            # whether to plot the training curve
                clip_grad_norm=False):
    print("training with clip_grad_norm=", clip_grad_norm)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True) # reshuffle minibatches every epoch
    criterion = nn.CrossEntropyLoss()
    if weight_decay > 0:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # these lists will be used to track the training progress
    # and to plot the training curve
    iters, train_loss, train_acc, val_acc = [], [], [], []
    iter_count = 0 # count the number of iterations that has passed

    try:
        for e in range(num_epochs):
            for i, (texts, labels) in enumerate(train_loader):
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

            plt.show()


model = MyRNN(
    num_cols=13,
    hidden_size=15,
    num_classes=5
)

train_model(model, train_data, val_data,
            learning_rate=0.005, batch_size=10,
            num_epochs=100, plot_every=100, plot=True, clip_grad_norm=True)

def plot_incorrect_classifications(model, dataset):
    dataloader = DataLoader(dataset, batch_size=1)
    incorrect_classifications = np.zeros(len(global_constants.ASTHMA_CASES))
    model.eval()

    category_names = sorted(global_constants.ASTHMA_CASES, key=global_constants.ASTHMA_CASES.get)


    for texts, labels in dataloader:
        z = model(texts)
        predictions = torch.argmax(z, axis=1)
        labels = torch.argmax(labels, axis=1)
        for i in range(predictions.size(0)):
            if predictions[i] != labels[i]:
                incorrect_classifications[labels[i].item()] += 1

    plt.figure()
    categories = np.arange(len(category_names))
    plt.bar(categories, incorrect_classifications, color='red')
    plt.xlabel("Categories")
    plt.ylabel("Number of Incorrect Classifications")
    plt.title("Incorrect Classifications by Category")
    plt.xticks(categories, category_names)
    plt.show()


plot_incorrect_classifications(model, test_data)


# Count of patient severity levels in each dataset
def count_patients(dataset):
    dataloader = DataLoader(dataset, batch_size=1)
    severity_labels = ['none', 'mild', 'moderate', 'severe', 'life-threatening']
    severity_counts = {}
    
    for _, severity_tensor in dataloader:
        severity = severity_labels[torch.argmax(severity_tensor).item()]
        if severity in severity_counts:
            severity_counts[severity] += 1
        else:
            severity_counts[severity] = 1
            
    return severity_counts


train_counts = count_severity_patients(train_data)
val_counts = count_severity_patients(val_data)
test_counts = count_severity_patients(test_data)

print("Training Set:")
print("--------------")
for severity, count in train_counts.items():
    print(f"Severity: {severity}, Patients: {count}")
print("--------------")

print("Validation Set:")
print("--------------")
for severity, count in val_counts.items():
    print(f"Severity: {severity}, Patients: {count}")
print("--------------")

print("Test Set:")
print("--------------")
for severity, count in test_counts.items():
    print(f"Severity: {severity}, Patients: {count}")
print("--------------")





