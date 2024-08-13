import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.init import xavier_uniform_, kaiming_uniform_

import time
import numpy as np
from tqdm import tqdm
from typing import Literal
from utils import EarlyStopping

__model_version__ = '1.0'

class SentimentAnalyzer(nn.Module):
    '''
    Sentiment Analysis Model with LSTM, GRU, RNN, or CNN layers.
    '''
    def __init__(self, model_type: Literal['lstm', 'gru', 'rnn', 'cnn'], vocab_size: int, embedding_dim: int, hidden_dim: int, pad_idx: int, bidirectional: bool = False, n_layers: int = None, dropout=0.2, lr=1e-3, weight_decay=1e-5, verbose=False, gpu=False):
        super(SentimentAnalyzer, self).__init__()

        self.model_type = model_type

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # RNN/CNN layer
        if model_type == 'lstm':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        elif model_type == 'gru':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        elif model_type == 'rnn':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        else:
            self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_dim//3, kernel_size=3)
            self.conv2 = nn.Conv1d(in_channels=hidden_dim//3, out_channels=hidden_dim//2, kernel_size=3)
            self.conv3 = nn.Conv1d(in_channels=hidden_dim//2, out_channels=hidden_dim, kernel_size=3)
            self.pool = nn.MaxPool1d(kernel_size=2)
            self.relu = nn.ReLU()

        # Output layer
        if bidirectional:
            out_size = 2 * hidden_dim
        elif model_type == 'cnn':
            out_size = 4480 # Size after flattening the output of the convolutional/pooling layers (Could be calculated dynamically)
        else:
            out_size = hidden_dim

        # Output layer
        self.fc = nn.Linear(out_size, 1)

        self.sigmoid = nn.Sigmoid()

        self.criterion = nn.BCELoss() # Binary classification
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay) # weight_decay is L2 regularization
        self.device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
        
        self.__init_weights()

        self.to(self.device)

        # Print the model architecture
        print(self, '\nRunning on: ', self.device) if verbose else None
        print(f'Number of parameters: {self.params_count():,}') if verbose else None

    def __init_data(self, X: np.ndarray, y: np.ndarray = None) -> tuple[torch.IntTensor, torch.FloatTensor]:
        '''
        Initialize the data
        '''
        X = torch.IntTensor(X).to(self.device, non_blocking=True)
        y = torch.FloatTensor(y).to(self.device, non_blocking=True) if y is not None else None

        return X, y

    def __init_weights(self):
        # Initialize embedding layer
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

        # Initialize RNN layer weights
        if self.model_type != 'cnn':
            for name, param in self.rnn.named_parameters():
                if 'weight_ih' in name: # input weights
                    xavier_uniform_(param.data)
                elif 'weight_hh' in name: # hidden weights
                    xavier_uniform_(param.data)
                elif 'bias' in name: # bias
                    param.data.fill_(0)
                    if self.model_type == 'lstm': # Set forget gate bias to 1 for LSTM (To encourage learning long-term dependencies from the start)
                        num_biases = param.size(0)
                        param.data[num_biases//4:num_biases//2].fill_(1)
        else:
            # Initialize convolutional layers
            kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
            nn.init.constant_(self.conv1.bias, 0)
            kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
            nn.init.constant_(self.conv2.bias, 0)
            kaiming_uniform_(self.conv3.weight, nonlinearity='relu')
            nn.init.constant_(self.conv3.bias, 0)

        # Initialize fully connected layer
        xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass
        '''
        embedded = self.embedding(text)  # (batch_size, seq_len, embedding_dim)
        if self.model_type != 'cnn':
            output, hidden = self.rnn(embedded)  # hidden: (num_layers * num_directions, batch_size, hidden_dim)
            if self.model_type == 'lstm':
                hidden = hidden[0]  # hidden contains (h_n, c_n) for LSTM
            if self.rnn.bidirectional:
                hidden = hidden.view(self.rnn.num_layers, 2, -1, self.rnn.hidden_size)
                hidden = torch.cat((hidden[-1, 0], hidden[-1, 1]), dim=1)
            else:
                hidden = hidden[-1, :, :]
            output = self.fc(hidden)  # (batch_size, 1)
        else:
            embedded = embedded.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)
            x = self.pool(self.relu(self.conv1(embedded)))  # (batch_size, hidden_dim//3, seq_len-2)
            x = self.pool(self.relu(self.conv2(x)))  # (batch_size, hidden_dim//2, seq_len-4)
            x = self.pool(self.relu(self.conv3(x)))  # (batch_size, hidden_dim, seq_len-6)
            output = self.fc(x.view(x.size(0), -1)) # (batch_size, 1)

        return self.sigmoid(output.squeeze(-1))

    def fit(self, X: np.ndarray, y: np.ndarray, batch_size: int, epochs: int, scheduler: torch.optim.lr_scheduler, early_stopping: EarlyStopping, val_data: tuple[np.ndarray, np.ndarray] = None) -> dict:
        '''
        Train the model

        params:
        X: np.ndarray - Input features
        y: np.ndarray - Target values
        batch_size: int - Batch size
        epochs: int - Number of epochs
        val_data: tuple[np.ndarray, np.ndarray] - Validation data
        scheduler: torch.optim.lr_scheduler - Learning rate scheduler
        early_stopping: EarlyStopping - Early stopping object

        returns:
        dict - Training history
        '''
        X, y = self.__init_data(X, y)

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        self.train()

        # self.__init_weights() # Reinitialize the weights to prevent using the weights from the previous training session

        losses = []
        all_metrics = []
        epoch_time = 0

        for epoch in range(epochs):
            start_time = time.time()
            total_loss = 0.0
            with tqdm(dataloader, unit='batch', desc=f'Epoch {epoch+1}/{epochs}', leave=False) as tepoch:
                for i, batch in enumerate(tepoch):
                    X, y = batch

                    self.optimizer.zero_grad() # Zero the gradients
                    output = self(X) # Forward pass
                    loss = self.criterion(output, y) # Compute the loss
                    loss.backward() # Backward pass
                    self.optimizer.step() # Update the weights

                    total_loss += loss.item()
                    
                    tepoch.set_postfix(loss=(total_loss / (i+1)))

            epoch_time = max(epoch_time, time.time() - start_time) # Get the maximum time taken for an epoch

            avg_loss = total_loss / len(dataloader)
            losses.append(avg_loss)
            
            # Evaluate the model
            if val_data:
                self.eval()
                metrics = self.evaluate(val_data, batch_size)

                if scheduler:
                    scheduler.step(metrics[0])

                last_lr = scheduler.optimizer.param_groups[0]['lr']
                all_metrics.append([*metrics] + [last_lr])
                tqdm.write(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f} - Val Loss: {metrics[0]:.4f} - Acc: {metrics[1]:.4f} - F1: {metrics[2]:.4f} - Precision: {metrics[3]:.4f} - Recall: {metrics[4]:.4f} - lr: {last_lr}')
                
                if early_stopping:
                    early_stopping(metrics[0], self)
                    if early_stopping.early_stop:
                        print("Early stopping...")
                        break

                self.train()
            else:
                tqdm.write(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}')

        history = {
            'loss': losses,
            'val_loss': [m[0] for m in all_metrics],
            'acc': [m[1] for m in all_metrics],
            'f1': [m[2] for m in all_metrics],
            'precision': [m[3] for m in all_metrics],
            'recall': [m[4] for m in all_metrics],
            'lr': [m[5] for m in all_metrics],
            'epoch_time': epoch_time
        } if val_data else {
            'loss': losses,
            'epoch_time': epoch_time
        }

        return history

    def predict(self, text: np.ndarray, batch_size: int=256) -> np.ndarray:
        '''
        Alias for forward method, Initializes data first then calls forward method
        '''
        text, _ = self.__init_data(text)
        preds = []
        with torch.no_grad():
            for i in range(0, len(text), batch_size):
                X = text[i:i+batch_size]
                preds.extend(self(X).cpu().numpy())
        return np.array(preds).reshape(-1)

    def evaluate(self, val_data: tuple[np.ndarray, np.ndarray], batch_size: int) -> tuple[float]:
        '''
        Evaluate the model

        params:
        val_data: tuple[np.ndarray, np.ndarray] - Validation data
        batch_size: int - Batch size

        returns:
        tuple[float] - Evaluation metrics (loss, accuracy, f1, precision, recall)
        '''
        X_val, y_val = self.__init_data(val_data[0], val_data[1])

        dataset = TensorDataset(X_val, y_val)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

        avg_loss, y_preds = 0.0, []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                X, y = batch

                preds = self(X)
                y_preds.extend(preds.cpu().numpy())

                # Compute loss
                loss = self.criterion(preds, y)
                avg_loss += loss.item()

        avg_loss /= (i + 1)

        target = y_val.cpu().numpy()
        y_preds = np.array(y_preds).reshape(-1)

        # Compute additional evaluation metrics
        tp = np.sum((y_preds.round() == 1) & (target == 1))
        fp = np.sum((y_preds.round() == 1) & (target == 0))
        fn = np.sum((y_preds.round() == 0) & (target == 1))

        acc = (y_preds.round() == target).mean()
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        return avg_loss, acc, f1, precision, recall

    def params_count(self)-> int:
        '''
        Count the number of parameters in the model
        '''
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def memory_usage(self)-> float:
        '''
        Calculate the memory usage of the model
        '''
        # Calculate memory for parameters
        param_memory = sum(p.numel() for p in self.parameters()) * 4  # Assuming float32, which is 4 bytes

        # Calculate memory for gradients (same size as parameters)
        grad_memory = sum(p.numel() for p in self.parameters() if p.requires_grad) * 4  # Only if gradients are required

        # Calculate memory for buffers
        buffer_memory = sum(b.numel() for b in self.buffers()) * 4

        # Total memory
        total_memory = param_memory + grad_memory + buffer_memory

        # Convert bytes to megabytes for easier reading
        return total_memory / (1024 ** 2)
    
    def save_weights(self, path)-> None:
        '''
        Save the model weights
        '''
        torch.save(self.state_dict(), path)

    def load_weights(self, path, eval=True)-> None:
        '''
        Load the model weights

        Note: Set eval to True if you want to set the model to evaluation mode
        '''
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.eval() if eval else None 