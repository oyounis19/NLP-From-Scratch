import torch
from torch.utils.data import Dataset
from typing import Literal, Tuple

class TextDataset(Dataset):
    def __init__(
            self, 
            data: list[Tuple], 
            task: Literal['classification', 'generation', 'translation'], 
            tokenizer, 
            max_length: int = 512
            ):
        '''
        Custom dataset for different NLP tasks.
        
        :param data: List of tuples, of input and target text (label for classification).
        :param task: String, one of ['classification', 'generation', 'translation'].
        :param tokenizer: Tokenizer object to convert text to input_ids.
        :param max_length: Integer, maximum length of input_ids.
        '''
        self.data = data
        self.task = task
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_text, target = self.data[idx]
        input_ids = self.tokenizer.encode(input_text, max_length=self.max_length, padding='max_length', truncation=True)
        attention_mask = [1 if i != self.tokenizer.pad_token_id else 0 for i in input_ids]
        
        if self.task == 'classification':
            target = torch.tensor(target)
            return {
                'input_ids': torch.tensor(input_ids),
                'attention_mask': torch.tensor(attention_mask),
                'target': target
            }
        else:
            return {
                'input_ids': torch.tensor(input_ids),
                'attention_mask': torch.tensor(attention_mask),
                'target': self.tokenizer.encode(target, max_length=self.max_length, padding='max_length', truncation=True)
            }