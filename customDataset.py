import torch
from torch.utils.data import Dataset 

class customDataset(Dataset):

    def __init__(self, sentences, tokenizer, gpt2_type="gpt2", max_length=1024):

        self.tokenizer = tokenizer 
        self.input_ids = []
        self.attn_masks = []

        for sentence in sentences:      
            encodings = self.tokenize_seq(sentence,tokenizer,max_length)
                    
            self.input_ids.append(torch.tensor(encodings['input_ids']))
            self.attn_masks.append(torch.tensor(encodings['attention_mask']))
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx] 

    def tokenize_seq(self, sent,tokenizer,max_length):
        return tokenizer(sent, truncation=True, max_length=min(max_length, 1024), padding="max_length")
