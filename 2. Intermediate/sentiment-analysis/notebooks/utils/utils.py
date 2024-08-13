import torch
import torch.nn as nn

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tabulate import tabulate
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix

class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after a certain number of epochs (patience).
    """
    def __init__(self, patience=3, delta=0, verbose=False, path='checkpoint.pth') -> None:
        """
        Args:
            patience (int): How many epochs to wait after the last time the validation loss improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            verbose (bool): If True, prints a message for each validation loss improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss, model: nn.Module) -> None:
        '''
        Call method
        '''
        score = -val_loss

        if not self.best_score:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}') if self.verbose else None
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model: nn.Module) -> None:
        '''
        Save the model checkpoint
        '''
        print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...') if self.verbose else None
        torch.save(model.state_dict(), self.path)
        self.best_loss = val_loss

class Utils:
    @staticmethod
    def plot_training_history(histories):
        '''
        Plot all metrics for all models
        '''
        fig, axs = plt.subplots(2, 3, figsize=(15, 8))
        axs = axs.ravel()
        metrics = ['loss', 'val_loss', 'acc', 'f1', 'precision', 'recall']
        
        for i, metric in enumerate(metrics):
            for model, history in histories.items():
                axs[i].plot(history[metric], label=model)
            axs[i].set_title(metric.capitalize())
            axs[i].set_xlabel('Epochs')
            axs[i].set_ylabel(metric.capitalize())
            axs[i].legend()
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_confusion_matrices(true_labels, models_preds):
        '''
        Plot confusion matrices for all models
        '''
        _, axs = plt.subplots(2, 4, figsize=(20, 10))
        axs = axs.flatten()
        for i, (model_name, predictions) in enumerate(models_preds.items()):
                # Convert continuous predictions to discrete class labels
                predicted_labels = (predictions > 0.5).astype(int) 

                # Compute the confusion matrix
                cm = confusion_matrix(true_labels, predicted_labels)

                # Plot the confusion matrix using seaborn
                sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', ax=axs[i])
                axs[i].set_title(f'{model_name}')
                axs[i].set_xlabel('Predicted')
                axs[i].set_ylabel('True')
        axs[-1].set_visible(False)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def generate_comparison_table(histories: dict, models: dict) -> None:
        '''
        Generate a comparison table for all models
        '''
        # sort the histories by the validation loss
        histories = {k: v for k, v in sorted(histories.items(), key=lambda item: min(item[1]['val_loss']))}

        data = [
            {
                'Model': model,
                'Train Loss': round(history['loss'][-1], 4),
                'Val Loss': round(history['val_loss'][-1], 4),
                'Accuracy': round(history['acc'][-1], 4),
                'Precision': round(history['precision'][-1], 4),
                'F1-Score': round(history['f1'][-1], 4),
                'Recall': round(history['recall'][-1], 4),
                'Time per Epoch (s)': round(history['epoch_time']),
                'Training Time (Min)': round(history['epoch_time'] * len(history['loss']) / 60, 2),
                'Memory Usage (MB)': round(models[model].memory_usage()),
                'Parameters': f'{models[model].params_count():,}'
            }
            for model, history in histories.items()
        ]

        df = pd.DataFrame(data)
        print(tabulate(df, headers='keys', tablefmt='pretty', showindex=False))

    @staticmethod
    def tokenize_and_preprocess(text, word_to_idx, max_len) -> list:
        '''
        Tokenize and preprocess the text
        '''
        # Tokenize the text
        tokens = word_tokenize(text.lower())
        
        # Convert tokens to indices
        tokens = [word_to_idx[token] for token in tokens if token in word_to_idx]
        
        # Pad the tokens to the max_len
        tokens += [0] * (max_len - len(tokens))

        return tokens