import numpy as np
import torch
import pandas as pd
# from torch.utils.data import Dataset


class Dataset(torch.utils.data.Dataset):
    def __init__(self,file_path, tokenizer, mode='train'):
        self.file_path = file_path
        self.mode = mode
        self.tokenizer = tokenizer
        self.df = pd.read_csv(file_path)
        
        self.data = self.preprocess()
        
    def preprocess(self):
        self.df['token'] = self.df['token'].map(lambda x: eval(x))
        self.df['ner_label'] = self.df['ner_label'].map(lambda x: eval(x))
        
        return {
            "token": self.df['token'].tolist(),
            "ner_label": self.df['ner_label'].tolist()
        }
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        return self.data['token'][index], self.data['ner_label'][index]
    def align_labels_with_tokens(self, labels, word_ids):
        previous_word_id = None
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(0)
            elif word_id != previous_word_id:
                label_ids.append(labels[word_id])
            else:
                label_ids.append(0)
            previous_word_id = word_id
        return label_ids
    
    def tokenize_and_align_labels(self,tokens, labels):
        tokenized_inputs = self.tokenizer(
            tokens,
            padding="longest",
            truncation=True,
            is_split_into_words=True,
            return_tensors="pt"
        )
        
        labels_align = []
        for i, label in enumerate(labels):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            labels_align.append(self.align_labels_with_tokens(label, word_ids))
        
        tokenized_inputs["labels"] = torch.from_numpy(np.array(labels_align)).long()
        return tokenized_inputs
    def data_collator(self, batch):
        tokens = [x[0] for x in batch]
        labels = [x[1] for x in batch]

        model_inputs = self.tokenize_and_align_labels(tokens, labels)
        model_inputs['attention_mask'] = torch.from_numpy(np.array(model_inputs['attention_mask'])).long()

        return model_inputs