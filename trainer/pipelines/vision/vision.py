import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from trainer.dataloader.vision_dataloader import MNISTCsvDataset
import matplotlib.pyplot as plt

from trainer.model.vision.model import SimpleMLP

train_ds = MNISTCsvDataset('data/mnist_train.csv')
test_ds = MNISTCsvDataset('data/mnist_test.csv')


history_losses = np.zeros(len(train_ds))
MOVING_AVG_DECAY = 0.9

def get_smart_batch(loss_history, batch_size, explore_frac=0.5, top_k_frac=0.2):
    n_explore = int(batch_size * explore_frac)
    n_exploit = batch_size - n_explore
    n_total = len(loss_history)

    rand_idxs = np.random.choice(n_total, n_explore, replace=False)

    k = int(top_k_frac * n_total)
    exploit_candidates = np.argsort(-loss_history)[:k]
    if len(exploit_candidates) < n_exploit:
        exploit_idxs = exploit_candidates
    else:
        exploit_idxs = np.random.choice(exploit_candidates, n_exploit, replace=False)

    batch_idxs = np.concatenate([rand_idxs, exploit_idxs])
    np.random.shuffle(batch_idxs)
    return batch_idxs


def train_random(model, train_ds, test_ds, epochs=5, batch_size=64):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    train_accuracies, test_accuracies, train_losses = [], [], []
    for epoch in range(epochs):
        correct, n, running_loss = 0, 0, 0
        model.train()
        for x, y in train_loader:
            x, y = x.to('cpu'), y.to('cpu')
            x = x.view(x.size(0), -1)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            pred_labels = y_pred.argmax(1)
            correct += (pred_labels == y).sum().item()
            n += x.size(0)
        train_accuracies.append(correct / n)
        train_losses.append(running_loss / n)

        test_acc = evaluate(model, test_ds)
        test_accuracies.append(test_acc)
        print(f"Rand-Epoch {epoch+1}: train_acc={train_accuracies[-1]:.4f}, test_acc={test_acc:.4f}")
    return train_accuracies, test_accuracies, train_losses



def train_smart(model, train_ds, test_ds, epochs=5, batch_size=64):
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters())

    per_sample_loss = np.zeros(len(train_ds))
    train_accuracies, test_accuracies, train_losses = [], [], []

    for epoch in range(epochs):
        perm = np.arange(len(train_ds))
        n_batches = len(train_ds) // batch_size
        correct, n, running_loss = 0, 0, 0
        model.train()
        for i in range(n_batches):
            idxs = get_smart_batch(per_sample_loss, batch_size)
            x, y = zip(*[train_ds[j] for j in idxs])
            x = torch.stack(x).view(batch_size, -1).to('cpu')
            y = torch.tensor(y).to('cpu')
            y_pred = model(x)
            losses = loss_fn(y_pred, y)
            loss = losses.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            pred_labels = y_pred.argmax(1)
            correct += (pred_labels == y).sum().item()
            n += x.size(0)

            for k, idx in enumerate(idxs):
                per_sample_loss[idx] = MOVING_AVG_DECAY * per_sample_loss[idx] + (1-MOVING_AVG_DECAY) * losses[k].item()

        train_accuracies.append(correct / n)
        train_losses.append(running_loss / n)

        test_acc = evaluate(model, test_ds)
        test_accuracies.append(test_acc)
        print(f"Smart-Epoch {epoch+1}: train_acc={train_accuracies[-1]:.4f}, test_acc={test_acc:.4f}")
    return train_accuracies, test_accuracies, train_losses



def evaluate(model, ds):
    loader = DataLoader(ds, batch_size=256)
    model.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.view(x.size(0), -1).to('cpu')
            y = y.to('cpu')
            y_pred = model(x)
            pred_labels = y_pred.argmax(1)
            correct += (pred_labels == y).sum().item()
            n += x.size(0)
    return correct / n




rand_model = SimpleMLP()
rand_train_acc, rand_test_acc, rand_train_loss = train_random(rand_model, train_ds, test_ds)

smart_model = SimpleMLP()
smart_train_acc, smart_test_acc, smart_train_loss = train_smart(smart_model, train_ds, test_ds)

epochs = range(1, len(rand_train_acc)+1)
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(epochs, rand_train_acc, label='Random Train Acc')
plt.plot(epochs, smart_train_acc, label='Smart Train Acc')
plt.plot(epochs, rand_test_acc, label='Random Test Acc')
plt.plot(epochs, smart_test_acc, label='Smart Test Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Comparison')

plt.subplot(1,2,2)
plt.plot(epochs, rand_train_loss, label='Random Train Loss')
plt.plot(epochs, smart_train_loss, label='Smart Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Comparison')

plt.tight_layout()
plt.show()