"""
Script to compute data statistics for Table 1 and Table 2 in the report.
"""
import os
from collections import Counter
from transformers import T5TokenizerFast
from load_data import load_lines

def compute_statistics(data_folder='data'):
    """
    Compute statistics for train and dev sets before and after preprocessing.
    """
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    # Load data
    train_nl = load_lines(os.path.join(data_folder, 'train.nl'))
    train_sql = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_nl = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_sql = load_lines(os.path.join(data_folder, 'dev.sql'))
    
    print("=" * 80)
    print("Table 1: Data statistics before any pre-processing")
    print("=" * 80)
    
    # Statistics before preprocessing (using raw text)
    def compute_before_stats(nl_lines, sql_lines):
        # Number of examples
        num_examples = len(nl_lines)
        
        # Mean sentence length (in words, before tokenization)
        nl_lengths = [len(line.split()) for line in nl_lines]
        mean_nl_length = sum(nl_lengths) / len(nl_lengths) if nl_lengths else 0
        
        # Mean SQL query length (in words, before tokenization)
        sql_lengths = [len(line.split()) for line in sql_lines]
        mean_sql_length = sum(sql_lengths) / len(sql_lengths) if sql_lengths else 0
        
        # Vocabulary size (natural language) - unique words
        nl_vocab = set()
        for line in nl_lines:
            words = line.lower().split()
            nl_vocab.update(words)
        nl_vocab_size = len(nl_vocab)
        
        # Vocabulary size (SQL) - unique words
        sql_vocab = set()
        for line in sql_lines:
            words = line.split()
            sql_vocab.update(words)
        sql_vocab_size = len(sql_vocab)
        
        return {
            'num_examples': num_examples,
            'mean_nl_length': mean_nl_length,
            'mean_sql_length': mean_sql_length,
            'nl_vocab_size': nl_vocab_size,
            'sql_vocab_size': sql_vocab_size
        }
    
    train_stats_before = compute_before_stats(train_nl, train_sql)
    dev_stats_before = compute_before_stats(dev_nl, dev_sql)
    
    print(f"Number of examples: Train={train_stats_before['num_examples']}, Dev={dev_stats_before['num_examples']}")
    print(f"Mean sentence length: Train={train_stats_before['mean_nl_length']:.2f}, Dev={dev_stats_before['mean_nl_length']:.2f}")
    print(f"Mean SQL query length: Train={train_stats_before['mean_sql_length']:.2f}, Dev={dev_stats_before['mean_sql_length']:.2f}")
    print(f"Vocabulary size (natural language): Train={train_stats_before['nl_vocab_size']}, Dev={dev_stats_before['nl_vocab_size']}")
    print(f"Vocabulary size (SQL): Train={train_stats_before['sql_vocab_size']}, Dev={dev_stats_before['sql_vocab_size']}")
    
    print("\n" + "=" * 80)
    print("Table 2: Data statistics after pre-processing (using T5 tokenizer)")
    print("=" * 80)
    
    # Statistics after preprocessing (using T5 tokenizer)
    def compute_after_stats(nl_lines, sql_lines, tokenizer):
        # Mean sentence length (in tokens after tokenization)
        nl_token_lengths = []
        for line in nl_lines:
            tokens = tokenizer.encode(line, add_special_tokens=False)
            nl_token_lengths.append(len(tokens))
        mean_nl_token_length = sum(nl_token_lengths) / len(nl_token_lengths) if nl_token_lengths else 0
        
        # Mean SQL query length (in tokens after tokenization)
        sql_token_lengths = []
        for line in sql_lines:
            tokens = tokenizer.encode(line, add_special_tokens=False)
            sql_token_lengths.append(len(tokens))
        mean_sql_token_length = sum(sql_token_lengths) / len(sql_token_lengths) if sql_token_lengths else 0
        
        # Vocabulary size (natural language) - unique token IDs
        nl_token_vocab = set()
        for line in nl_lines:
            tokens = tokenizer.encode(line, add_special_tokens=False)
            nl_token_vocab.update(tokens)
        nl_token_vocab_size = len(nl_token_vocab)
        
        # Vocabulary size (SQL) - unique token IDs
        sql_token_vocab = set()
        for line in sql_lines:
            tokens = tokenizer.encode(line, add_special_tokens=False)
            sql_token_vocab.update(tokens)
        sql_token_vocab_size = len(sql_token_vocab)
        
        return {
            'mean_nl_token_length': mean_nl_token_length,
            'mean_sql_token_length': mean_sql_token_length,
            'nl_token_vocab_size': nl_token_vocab_size,
            'sql_token_vocab_size': sql_token_vocab_size
        }
    
    train_stats_after = compute_after_stats(train_nl, train_sql, tokenizer)
    dev_stats_after = compute_after_stats(dev_nl, dev_sql, tokenizer)
    
    print(f"Mean sentence length (tokens): Train={train_stats_after['mean_nl_token_length']:.2f}, Dev={dev_stats_after['mean_nl_token_length']:.2f}")
    print(f"Mean SQL query length (tokens): Train={train_stats_after['mean_sql_token_length']:.2f}, Dev={dev_stats_after['mean_sql_token_length']:.2f}")
    print(f"Vocabulary size (natural language, token IDs): Train={train_stats_after['nl_token_vocab_size']}, Dev={dev_stats_after['nl_token_vocab_size']}")
    print(f"Vocabulary size (SQL, token IDs): Train={train_stats_after['sql_token_vocab_size']}, Dev={dev_stats_after['sql_token_vocab_size']}")
    
    print("\n" + "=" * 80)
    print("LaTeX Table Format:")
    print("=" * 80)
    print("\nTable 1:")
    print(f"Number of examples & {train_stats_before['num_examples']} & {dev_stats_before['num_examples']} \\\\")
    print(f"Mean sentence length & {train_stats_before['mean_nl_length']:.2f} & {dev_stats_before['mean_nl_length']:.2f} \\\\")
    print(f"Mean SQL query length & {train_stats_before['mean_sql_length']:.2f} & {dev_stats_before['mean_sql_length']:.2f} \\\\")
    print(f"Vocabulary size (natural language) & {train_stats_before['nl_vocab_size']} & {dev_stats_before['nl_vocab_size']} \\\\")
    print(f"Vocabulary size (SQL) & {train_stats_before['sql_vocab_size']} & {dev_stats_before['sql_vocab_size']} \\\\")
    
    print("\nTable 2:")
    print(f"Mean sentence length (tokens) & {train_stats_after['mean_nl_token_length']:.2f} & {dev_stats_after['mean_nl_token_length']:.2f} \\\\")
    print(f"Mean SQL query length (tokens) & {train_stats_after['mean_sql_token_length']:.2f} & {dev_stats_after['mean_sql_token_length']:.2f} \\\\")
    print(f"Vocabulary size (natural language, token IDs) & {train_stats_after['nl_token_vocab_size']} & {dev_stats_after['nl_token_vocab_size']} \\\\")
    print(f"Vocabulary size (SQL, token IDs) & {train_stats_after['sql_token_vocab_size']} & {dev_stats_after['sql_token_vocab_size']} \\\\")

if __name__ == "__main__":
    compute_statistics()

