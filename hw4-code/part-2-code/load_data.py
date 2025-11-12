import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.split = split
        self.data = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        '''
        Load and process the data for the given split.
        For train/dev: load both NL and SQL
        For test: load only NL
        '''
        nl_path = os.path.join(data_folder, f'{split}.nl')
        nl_lines = load_lines(nl_path)
        
        if split == 'test':
            # For test set, we only have natural language queries
            return [(nl, None) for nl in nl_lines]
        else:
            # For train/dev, we have both NL and SQL
            sql_path = os.path.join(data_folder, f'{split}.sql')
            sql_lines = load_lines(sql_path)
            return list(zip(nl_lines, sql_lines))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        nl_query, sql_query = self.data[idx]
        
        # Tokenize encoder input (natural language)
        encoder_inputs = self.tokenizer(
            nl_query,
            padding=False,
            truncation=True,
            max_length=512,
            return_tensors=None
        )
        encoder_ids = encoder_inputs['input_ids']
        
        if self.split == 'test':
            # For test set, we don't have SQL targets
            return {
                'encoder_ids': encoder_ids,
                'decoder_ids': None,
                'decoder_targets': None
            }
        else:
            # Tokenize decoder input and targets (SQL)
            # Decoder input should start with a special token (using pad token as BOS)
            decoder_inputs = self.tokenizer(
                sql_query,
                padding=False,
                truncation=True,
                max_length=512,
                return_tensors=None
            )
            decoder_ids = decoder_inputs['input_ids']
            
            # For training, decoder input is the SQL sequence
            # Decoder target should be shifted by one position for teacher forcing
            # In T5, logits[i] predicts decoder_input_ids[i+1]
            # So targets should be decoder_ids[1:] (shifted)
            # But we'll keep the full sequence and handle alignment in loss computation
            decoder_targets = decoder_ids.copy()
            
            return {
                'encoder_ids': encoder_ids,
                'decoder_ids': decoder_ids,
                'decoder_targets': decoder_targets
            }

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids_list = [torch.tensor(item['encoder_ids'], dtype=torch.long) for item in batch]
    decoder_ids_list = [torch.tensor(item['decoder_ids'], dtype=torch.long) for item in batch]
    decoder_targets_list = [torch.tensor(item['decoder_targets'], dtype=torch.long) for item in batch]
    
    # Pad sequences
    encoder_ids = pad_sequence(encoder_ids_list, batch_first=True, padding_value=PAD_IDX)
    decoder_inputs = pad_sequence(decoder_ids_list, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(decoder_targets_list, batch_first=True, padding_value=PAD_IDX)
    
    # Create attention masks
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    # Initial decoder input is just the pad token (BOS token for T5)
    # T5 uses pad token (0) as the decoder start token
    initial_decoder_inputs = torch.zeros(len(batch), 1, dtype=torch.long)
    
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids_list = [torch.tensor(item['encoder_ids'], dtype=torch.long) for item in batch]
    
    # Pad sequences
    encoder_ids = pad_sequence(encoder_ids_list, batch_first=True, padding_value=PAD_IDX)
    
    # Create attention masks
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    # Initial decoder input is just the pad token (BOS token for T5)
    initial_decoder_inputs = torch.zeros(len(batch), 1, dtype=torch.long)
    
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x