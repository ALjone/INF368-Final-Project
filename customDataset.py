
import torch
print(1)
from torch.utils.data import Dataset 
print(2)
from transformers import GPT2Tokenizer
print(3)

class customDataset(Dataset):

    def __init__(self, sentences, tokenizer: GPT2Tokenizer, max_length=1024):

        self.tokenizer = tokenizer 
        #TODO torch tensor instead?
        self.input_ids = []
        self.attn_masks = []

        for sentence in sentences:      
            encodings = tokenizer(sentence, truncation=True, max_length=min(max_length, 1024), padding="max_length")
                    
            self.input_ids.append(torch.tensor(encodings['input_ids']))
            self.attn_masks.append(torch.tensor(encodings['attention_mask']))
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx] 
