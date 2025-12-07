import torch
from torch import nn
import torch.nn.functional as F
from trainer.constants import HIDDEN_DIM, NUM_CLASSES


class SimpleRNN(nn.Module):
    """Simple RNN model for sequence classification."""

    def __init__(self, input_size=1, hidden_size=HIDDEN_DIM, num_classes=NUM_CLASSES, batch_first=True):
        super().__init__()
        print(f'[model.py]: building an RNN with input_size={input_size}, hidden_size={hidden_size}, and num_classes={num_classes}')
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=batch_first)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, hidden=None):
        # x shape: (batch, seq_len, input_size) if batch_first=True
        #          (seq_len, batch, input_size) if batch_first=False
        rnn_out, hidden = self.rnn(x, hidden)

        # Take the output from the last time step
        if self.batch_first:
            out = rnn_out[:, -1, :]  # (batch, hidden_size)
        else:
            out = rnn_out[-1, :, :]  # (batch, hidden_size)

        # Pass through final linear layer
        out = self.fc(out)  # (batch, num_classes)
        return out


class SimpleLSTM(nn.Module):
    """Simple LSTM model for sequence classification."""

    def __init__(self, input_size=1, hidden_size=HIDDEN_DIM, num_classes=NUM_CLASSES, batch_first=True):
        super().__init__()
        print(f'[model.py]: building an LSTM with input_size={input_size}, hidden_size={hidden_size}, and num_classes={num_classes}')
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=batch_first)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, hidden=None):
        # x shape: (batch, seq_len, input_size) if batch_first=True
        #          (seq_len, batch, input_size) if batch_first=False
        lstm_out, hidden = self.lstm(x, hidden)

        # Take the output from the last time step
        if self.batch_first:
            out = lstm_out[:, -1, :]  # (batch, hidden_size)
        else:
            out = lstm_out[-1, :, :]  # (batch, hidden_size)

        # Pass through final linear layer
        out = self.fc(out)  # (batch, num_classes)
        return out

