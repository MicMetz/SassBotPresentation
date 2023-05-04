import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer



class TrainDataset(Dataset):
    # initialize basic structure, note saves both text and sarcasm content.
    def __init__(self, df, pretraine_path='xlm-roberta-base', max_length=128):
        self.df = df
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(pretraine_path)

    # prints standardized data content.
    def __getitem__(self, index):
        text = self.df.iloc[index]['text']
        label = self.df.iloc[index]["sarcastic"]

        encoded_input = self.tokenizer(
                text,
                max_length = self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt",
            )

        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"] if "attention_mask" in encoded_input else None

        data_input = {
            "input_ids":input_ids.flatten(),
            "attention_mask": attention_mask.flatten()
        }

        label_input ={
            "sarcasm": torch.tensor(label, dtype=torch.float),
        }

        return data_input, label_input

    def __len__(self):
        return self.df.shape[0]

# main test dataset class with only text process content.
class TestDataset(Dataset):
    def __init__(self, df, pretraine_path='xlm-roberta-base', max_length=128):
        self.df = df
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(pretraine_path)

    # item retrival.
    def __getitem__(self, index):
        text = self.df.iloc[index]["text"]

        encoded_input = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
        )

        # manages content with an attention mask on top.
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"] if "attention_mask" in encoded_input else None

        data_input = {
            "input_ids": input_ids.flatten(),
            "attention_mask": attention_mask.flatten()
        }

        return data_input

    def __len__(self):
        return self.df.shape[0]
