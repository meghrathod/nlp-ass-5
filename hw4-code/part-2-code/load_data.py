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
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.data = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        # Load natural language queries
        nl_path = os.path.join(data_folder, f'{split}.nl')
        nl_lines = load_lines(nl_path)
        
        data = []
        
        # For test set, we don't have SQL labels
        if split == 'test':
            for nl in nl_lines:
                # Add task prefix for T5 to understand the task
                input_text = f"translate to SQL: {nl}"
                
                # Tokenize the natural language input with task prefix
                encoder_input = tokenizer(input_text, return_tensors='pt', add_special_tokens=True)
                encoder_ids = encoder_input['input_ids'].squeeze(0)
                
                data.append({
                    'encoder_ids': encoder_ids,
                    'nl': nl
                })
        else:
            # Load SQL queries for train/dev
            sql_path = os.path.join(data_folder, f'{split}.sql')
            sql_lines = load_lines(sql_path)
            
            assert len(nl_lines) == len(sql_lines), f"Mismatch: {len(nl_lines)} NL vs {len(sql_lines)} SQL"
            
            for nl, sql in zip(nl_lines, sql_lines):
                # Add task prefix for T5 to understand the task
                input_text = f"translate to SQL: {nl}"
                
                # Tokenize the natural language input with task prefix
                encoder_input = tokenizer(input_text, return_tensors='pt', add_special_tokens=True)
                encoder_ids = encoder_input['input_ids'].squeeze(0)
                
                # Tokenize the SQL output
                decoder_output = tokenizer(sql, return_tensors='pt', add_special_tokens=True)
                decoder_ids = decoder_output['input_ids'].squeeze(0)
                
                data.append({
                    'encoder_ids': encoder_ids,
                    'decoder_ids': decoder_ids,
                    'nl': nl,
                    'sql': sql
                })
        
        return data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

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
    # Extract encoder and decoder sequences
    encoder_seqs = [item['encoder_ids'] for item in batch]
    decoder_seqs = [item['decoder_ids'] for item in batch]
    
    # Pad encoder sequences
    encoder_ids = pad_sequence(encoder_seqs, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    # For decoder, we need to create input/target pairs
    # decoder_input: starts with pad token (T5 uses pad as start token)
    # decoder_target: the actual sequence
    decoder_ids_padded = pad_sequence(decoder_seqs, batch_first=True, padding_value=PAD_IDX)
    
    # Decoder inputs: shift right (prepend PAD_IDX as start token)
    # T5 uses pad_token_id as the decoder_start_token_id
    decoder_inputs = torch.zeros((decoder_ids_padded.size(0), decoder_ids_padded.size(1)), dtype=torch.long)
    decoder_inputs[:, 0] = PAD_IDX  # Start with pad token
    decoder_inputs[:, 1:] = decoder_ids_padded[:, :-1]
    
    # Decoder targets: original sequences (shifted by decoder inputs)
    decoder_targets = decoder_ids_padded
    
    # Initial decoder input for generation (just the start token)
    initial_decoder_inputs = torch.full((len(batch), 1), PAD_IDX, dtype=torch.long)
    
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
    # Extract encoder sequences
    encoder_seqs = [item['encoder_ids'] for item in batch]
    
    # Pad encoder sequences
    encoder_ids = pad_sequence(encoder_seqs, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    # Initial decoder input for generation (just the start token)
    initial_decoder_inputs = torch.full((len(batch), 1), PAD_IDX, dtype=torch.long)
    
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split, num_workers=4):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(
        dset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,  # Faster data transfer to GPU
        persistent_workers=num_workers > 0  # Keep workers alive between epochs
    )
    return dataloader

def load_t5_data(batch_size, test_batch_size, num_workers=4):
    train_loader = get_dataloader(batch_size, "train", num_workers)
    dev_loader = get_dataloader(test_batch_size, "dev", num_workers)
    test_loader = get_dataloader(test_batch_size, "test", num_workers)
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x