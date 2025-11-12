import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def get_wordnet_pos(nltk_pos):
    """Convert NLTK POS tag to WordNet POS tag."""
    if nltk_pos.startswith('J'):
        return wordnet.ADJ
    elif nltk_pos.startswith('V'):
        return wordnet.VERB
    elif nltk_pos.startswith('N'):
        return wordnet.NOUN
    elif nltk_pos.startswith('R'):
        return wordnet.ADV
    else:
        return None


def preserve_capitalization(original_word, replacement_word):
    """Preserve the capitalization pattern of the original word."""
    if original_word.isupper():
        return replacement_word.upper()
    elif original_word.istitle():
        return replacement_word.capitalize()
    elif original_word[0].isupper() and len(original_word) > 1:
        # First letter uppercase, rest lowercase
        return replacement_word.capitalize()
    else:
        return replacement_word.lower()


# Domain-specific synonym groups - each set represents mutually synonymous words
# Using sets internally for automatic deduplication, then converting to lists for lookup
# Benefits: 1) No redundancy (define groups once), 2) Automatic bidirectional mapping,
#           3) Easy to maintain (add words to groups, not individual entries)
SYNONYM_GROUP_SETS = [
    # Movie/Film terms
    {'movie', 'film', 'picture', 'motion picture'},
    {'movies', 'films', 'pictures', 'motion pictures'},
    
    # Acting/Performance terms
    {'acting', 'performance', 'portrayal', 'playing'},
    {'actor', 'performer', 'player'},
    {'actors', 'performers', 'players'},
    {'actress', 'performer', 'actor'},
    {'actresses', 'performers', 'actors'},
    
    # Story/Narrative terms
    {'story', 'narrative', 'tale', 'plot'},
    {'stories', 'narratives', 'tales', 'plots'},
    {'screenplay', 'script'},
    
    # Character terms
    {'character', 'role', 'persona', 'part'},
    {'characters', 'roles', 'personas', 'parts'},
    
    # Quality/Evaluation terms (adjectives) - consolidated groups
    {'good', 'great', 'excellent', 'fine', 'nice', 'fantastic', 'wonderful', 'outstanding', 'superb', 'exceptional'},
    {'bad', 'terrible', 'awful', 'poor', 'horrible', 'dreadful'},
    {'amazing', 'awesome', 'incredible', 'fantastic', 'wonderful'},
    {'boring', 'dull', 'tedious', 'uninteresting'},
    {'funny', 'hilarious', 'amusing', 'comical'},
    {'sad', 'depressing', 'melancholic', 'heartbreaking'},
    
    # Common adverbs
    {'very', 'extremely', 'really', 'quite', 'incredibly'},
    
    # Common verbs
    {'watch', 'see', 'view'},
    {'like', 'enjoy', 'love', 'appreciate', 'adore'},
    {'hate', 'dislike', 'loathe', 'despise'},
    {'think', 'believe', 'feel', 'consider'},
]

# Build bidirectional lookup dictionary from groups
SYNONYM_GROUPS = {}
for group in SYNONYM_GROUP_SETS:
    for word in group:
        # Get all other words in the group as synonyms
        synonyms = list(group - {word})
        if word in SYNONYM_GROUPS:
            # Merge synonyms from multiple groups (if word appears in multiple)
            SYNONYM_GROUPS[word].extend(synonyms)
        else:
            SYNONYM_GROUPS[word] = synonyms
# Remove duplicates while preserving order
SYNONYM_GROUPS = {k: list(dict.fromkeys(v)) for k, v in SYNONYM_GROUPS.items()}


def analyze_word_frequency(dataset, top_n=100):
    """Analyze word frequency in dataset to identify common words."""
    from collections import Counter
    word_freq = Counter()
    for example in dataset:
        tokens = word_tokenize(example['text'].lower())
        word_freq.update(t for t in tokens if t.isalpha() and len(t) > 3)
    return word_freq.most_common(top_n)

def get_synonym_from_group(word):
    """Get synonym from domain-specific groups."""
    clean_word = re.sub(r'[^a-zA-Z]', '', word.lower())
    if clean_word in SYNONYM_GROUPS:
        alternatives = [s for s in SYNONYM_GROUPS[clean_word] if s.lower() != clean_word]
        return random.choice(alternatives) if alternatives else None
    return None

def get_wordnet_synonyms(word, pos_tag):
    """Get synonyms from WordNet as fallback."""
    wordnet_pos = get_wordnet_pos(pos_tag)
    if not wordnet_pos:
        return []
    clean_word = re.sub(r'[^a-zA-Z]', '', word.lower())
    if not clean_word:
        return []
    synsets = wordnet.synsets(clean_word, pos=wordnet_pos)
    synonyms = {lemma.name().replace('_', ' ').lower() 
                for synset in synsets for lemma in synset.lemmas()
                if lemma.name().replace('_', ' ').isalpha() 
                and lemma.name().replace('_', ' ').lower() != clean_word}
    return list(synonyms)

# Common verb tense transformations (present ↔ past, present ↔ past participle)
VERB_TENSE_MAP = {
    # Present to Past
    'is': 'was', 'are': 'were', 'am': 'was',
    'has': 'had', 'have': 'had',
    'do': 'did', 'does': 'did',
    'say': 'said', 'says': 'said',
    'go': 'went', 'goes': 'went',
    'get': 'got', 'gets': 'got',
    'make': 'made', 'makes': 'made',
    'know': 'knew', 'knows': 'knew',
    'think': 'thought', 'thinks': 'thought',
    'take': 'took', 'takes': 'took',
    'see': 'saw', 'sees': 'saw',
    'come': 'came', 'comes': 'came',
    'want': 'wanted', 'wants': 'wanted',
    'use': 'used', 'uses': 'used',
    'find': 'found', 'finds': 'found',
    'give': 'gave', 'gives': 'gave',
    'tell': 'told', 'tells': 'told',
    'work': 'worked', 'works': 'worked',
    'call': 'called', 'calls': 'called',
    'try': 'tried', 'tries': 'tried',
    'ask': 'asked', 'asks': 'asked',
    'need': 'needed', 'needs': 'needed',
    'feel': 'felt', 'feels': 'felt',
    'become': 'became', 'becomes': 'became',
    'leave': 'left', 'leaves': 'left',
    'put': 'put', 'puts': 'put',
    'mean': 'meant', 'means': 'meant',
    'keep': 'kept', 'keeps': 'kept',
    'let': 'let', 'lets': 'let',
    'begin': 'began', 'begins': 'began',
    'seem': 'seemed', 'seems': 'seemed',
    'help': 'helped', 'helps': 'helped',
    'show': 'showed', 'shows': 'showed',
    'hear': 'heard', 'hears': 'heard',
    'play': 'played', 'plays': 'played',
    'run': 'ran', 'runs': 'ran',
    'move': 'moved', 'moves': 'moved',
    'live': 'lived', 'lives': 'lived',
    'believe': 'believed', 'believes': 'believed',
    'bring': 'brought', 'brings': 'brought',
    'happen': 'happened', 'happens': 'happened',
    'write': 'wrote', 'writes': 'wrote',
    'sit': 'sat', 'sits': 'sat',
    'stand': 'stood', 'stands': 'stood',
    'lose': 'lost', 'loses': 'lost',
    'pay': 'paid', 'pays': 'paid',
    'meet': 'met', 'meets': 'met',
    'include': 'included', 'includes': 'included',
    'continue': 'continued', 'continues': 'continued',
    'set': 'set', 'sets': 'set',
    'learn': 'learned', 'learns': 'learned',
    'change': 'changed', 'changes': 'changed',
    'lead': 'led', 'leads': 'led',
    'understand': 'understood', 'understands': 'understood',
    'watch': 'watched', 'watches': 'watched',
    'follow': 'followed', 'follows': 'followed',
    'stop': 'stopped', 'stops': 'stopped',
    'create': 'created', 'creates': 'created',
    'speak': 'spoke', 'speaks': 'spoke',
    'read': 'read', 'reads': 'read',
    'spend': 'spent', 'spends': 'spent',
    'grow': 'grew', 'grows': 'grew',
    'open': 'opened', 'opens': 'opened',
    'walk': 'walked', 'walks': 'walked',
    'win': 'won', 'wins': 'won',
    'teach': 'taught', 'teaches': 'taught',
    'offer': 'offered', 'offers': 'offered',
    'remember': 'remembered', 'remembers': 'remembered',
    'consider': 'considered', 'considers': 'considered',
    'appear': 'appeared', 'appears': 'appeared',
    'buy': 'bought', 'buys': 'bought',
    'serve': 'served', 'serves': 'served',
    'die': 'died', 'dies': 'died',
    'send': 'sent', 'sends': 'sent',
    'build': 'built', 'builds': 'built',
    'stay': 'stayed', 'stays': 'stayed',
    'fall': 'fell', 'falls': 'fell',
    'cut': 'cut', 'cuts': 'cut',
    'reach': 'reached', 'reaches': 'reached',
    'kill': 'killed', 'kills': 'killed',
    'raise': 'raised', 'raises': 'raised',
    'pass': 'passed', 'passes': 'passed',
    'sell': 'sold', 'sells': 'sold',
    'decide': 'decided', 'decides': 'decided',
    'return': 'returned', 'returns': 'returned',
    'explain': 'explained', 'explains': 'explained',
    'develop': 'developed', 'develops': 'developed',
    'carry': 'carried', 'carries': 'carried',
    'break': 'broke', 'breaks': 'broke',
    'receive': 'received', 'receives': 'received',
    'agree': 'agreed', 'agrees': 'agreed',
    'support': 'supported', 'supports': 'supported',
    'hit': 'hit', 'hits': 'hit',
    'produce': 'produced', 'produces': 'produced',
    'eat': 'ate', 'eats': 'ate',
    'cover': 'covered', 'covers': 'covered',
    'catch': 'caught', 'catches': 'caught',
    'draw': 'drew', 'draws': 'drew',
    'choose': 'chose', 'chooses': 'chose',
    
    # Past to Present (reverse mapping)
    'was': 'is', 'were': 'are',
    'had': 'has',
    'did': 'does',
    'said': 'says',
    'went': 'goes',
    'got': 'gets',
    'made': 'makes',
    'knew': 'knows',
    'thought': 'thinks',
    'took': 'takes',
    'saw': 'sees',
    'came': 'comes',
    'wanted': 'wants',
    'used': 'uses',
    'found': 'finds',
    'gave': 'gives',
    'told': 'tells',
    'worked': 'works',
    'called': 'calls',
    'tried': 'tries',
    'asked': 'asks',
    'needed': 'needs',
    'felt': 'feels',
    'became': 'becomes',
    'left': 'leaves',
    'meant': 'means',
    'kept': 'keeps',
    'began': 'begins',
    'seemed': 'seems',
    'helped': 'helps',
    'showed': 'shows',
    'heard': 'hears',
    'played': 'plays',
    'ran': 'runs',
    'moved': 'moves',
    'lived': 'lives',
    'believed': 'believes',
    'brought': 'brings',
    'happened': 'happens',
    'wrote': 'writes',
    'sat': 'sits',
    'stood': 'stands',
    'lost': 'loses',
    'paid': 'pays',
    'met': 'meets',
    'included': 'includes',
    'continued': 'continues',
    'learned': 'learns',
    'changed': 'changes',
    'led': 'leads',
    'understood': 'understands',
    'watched': 'watches',
    'followed': 'follows',
    'stopped': 'stops',
    'created': 'creates',
    'spoke': 'speaks',
    'spent': 'spends',
    'grew': 'grows',
    'opened': 'opens',
    'walked': 'walks',
    'won': 'wins',
    'taught': 'teaches',
    'offered': 'offers',
    'remembered': 'remembers',
    'considered': 'considers',
    'appeared': 'appears',
    'bought': 'buys',
    'served': 'serves',
    'died': 'dies',
    'sent': 'sends',
    'built': 'builds',
    'stayed': 'stays',
    'fell': 'falls',
    'reached': 'reaches',
    'killed': 'kills',
    'raised': 'raises',
    'passed': 'passes',
    'sold': 'sells',
    'decided': 'decides',
    'returned': 'returns',
    'explained': 'explains',
    'developed': 'develops',
    'carried': 'carries',
    'broke': 'breaks',
    'received': 'receives',
    'agreed': 'agrees',
    'supported': 'supports',
    'produced': 'produces',
    'ate': 'eats',
    'covered': 'covers',
    'caught': 'catches',
    'drew': 'draws',
    'chose': 'chooses',
}

def change_verb_tense(word, pos_tag):
    """Change verb tense (present ↔ past) if possible."""
    # Only process verbs
    if not pos_tag.startswith('V'):
        return None
    
    clean_word = re.sub(r'[^a-zA-Z]', '', word.lower())
    if clean_word in VERB_TENSE_MAP:
        new_word = VERB_TENSE_MAP[clean_word]
        return preserve_capitalization(word, new_word)
    
    # Try to handle regular verbs ending in -ed (past) or -s/-es (present)
    # This is a simple heuristic and may not catch all cases
    if clean_word.endswith('ed') and len(clean_word) > 3:
        # Past tense -> Present tense (remove -ed, add -s or keep base form)
        base = clean_word[:-2]
        # Simple heuristic: if base form exists in map, use it
        if base + 's' in VERB_TENSE_MAP.values() or base in VERB_TENSE_MAP.values():
            # Try to find present form
            for present, past in VERB_TENSE_MAP.items():
                if past == clean_word:
                    return preserve_capitalization(word, present)
        # Otherwise, just remove -ed for base form (imperfect but reasonable)
        if len(base) > 2:
            return preserve_capitalization(word, base)
    
    # Handle present tense verbs ending in -s/-es -> base form or past
    if (clean_word.endswith('s') or clean_word.endswith('es')) and len(clean_word) > 3:
        base = clean_word[:-1] if clean_word.endswith('s') else clean_word[:-2]
        # Check if we have a past tense form
        if base in VERB_TENSE_MAP:
            past = VERB_TENSE_MAP[base]
            if past != clean_word:  # Only if it's different
                return preserve_capitalization(word, past)
    
    return None

# Contraction expansions and contractions
CONTRACTIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "can't": "cannot", "couldn't": "could not", "won't": "will not",
    "wouldn't": "would not", "shouldn't": "should not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "hasn't": "has not", "haven't": "have not", "hadn't": "had not",
    "it's": "it is", "that's": "that is", "what's": "what is",
    "there's": "there is", "here's": "here is", "where's": "where is",
    "i'm": "I am", "you're": "you are", "we're": "we are",
    "they're": "they are", "he's": "he is", "she's": "she is",
    "i've": "I have", "you've": "you have", "we've": "we have",
    "they've": "they have", "i'll": "I will", "you'll": "you will",
    "we'll": "we will", "they'll": "they will", "i'd": "I would",
    "you'd": "you would", "we'd": "we would", "they'd": "they would",
}

# Reverse mapping for contractions
CONTRACTIONS_REVERSE = {v: k for k, v in CONTRACTIONS.items()}

def expand_contractions(word):
    """Expand contractions if found."""
    clean_word = word.lower().strip(".,!?;:")
    if clean_word in CONTRACTIONS:
        expanded = CONTRACTIONS[clean_word]
        # Handle capitalization - if first letter is uppercase, capitalize first word of expansion
        if word and word[0].isupper():
            words = expanded.split()
            if words:
                words[0] = words[0].capitalize()
                return ' '.join(words)
        return expanded
    return None

def contract_expanded(word, next_word=None):
    """Contract expanded forms if possible."""
    # Check two-word combinations
    if next_word:
        two_words = f"{word.lower()} {next_word.lower()}"
        if two_words in CONTRACTIONS_REVERSE:
            contracted = CONTRACTIONS_REVERSE[two_words]
            if word[0].isupper():
                return contracted.capitalize()
            return contracted
    return None


def introduce_typo(word):
    """Introduce a typo in a word using QWERTY keyboard proximity."""
    if len(word) <= 2:
        return word, False
    
    # QWERTY keyboard proximity map (focusing on vowels as specified)
    qwerty_proximity = {
        'a': ['s', 'q', 'w', 'e'],
        'e': ['w', 'r', 'd', 's'],
        'i': ['u', 'o', 'k', 'j'],
        'o': ['i', 'p', 'l', 'k'],
        'u': ['y', 'i', 'j', 'h'],
        # Add some common consonants for better coverage
        's': ['a', 'w', 'e', 'd', 'x', 'z'],
        'd': ['s', 'e', 'r', 'f', 'c'],
        'r': ['e', 'd', 'f', 't'],
        't': ['r', 'f', 'g', 'y'],
        'n': ['b', 'h', 'j', 'm'],
        'l': ['k', 'o', 'p']
    }
    
    # Choose a random position (excluding first and last characters)
    if len(word) <= 3:
        return word, False
    
    pos = random.randint(1, len(word) - 2)
    char = word[pos].lower()
    
    if char in qwerty_proximity:
        replacement_char = random.choice(qwerty_proximity[char])
        # Preserve case
        if word[pos].isupper():
            replacement_char = replacement_char.upper()
        new_word = word[:pos] + replacement_char + word[pos + 1:]
        return new_word, True
    
    return word, False


def custom_transform(example):
    # Parameters - increased for more aggressive transformation
    p_synonym_adj, p_synonym_adv = 0.80, 0.80  # Increased further
    p_synonym_noun, p_synonym_verb = 0.65, 0.8  # Increased further
    p_tense_change = 0.75  # 50% chance to change verb tense
    p_contraction = 0.20  # 20% chance to expand/contract
    p_typo, min_word_length, max_typos_per_example = 0.40, 4, 10  # More typos
    
    tokens = word_tokenize(example["text"])
    pos_tags = pos_tag(tokens)
    
    new_tokens, typos_count = [], 0
    i = 0
    
    while i < len(tokens):
        token, pos = tokens[i], pos_tags[i][1]
        
        if not token.isalpha():
            new_tokens.append(token)
            i += 1
            continue
        
        wordnet_pos = get_wordnet_pos(pos)
        clean_word = re.sub(r'[^a-zA-Z]', '', token.lower())
        replaced = False
        
        # Stage 0: Contraction handling (check before other transformations)
        # Handle contractions that may have punctuation
        token_clean = token.strip(".,!?;:")
        if token_clean and random.random() < p_contraction:
            # Try to expand contraction
            expanded = expand_contractions(token_clean)
            if expanded:
                # Split into multiple tokens if needed
                expanded_tokens = word_tokenize(expanded)
                # Add punctuation back if it was stripped
                punct = token[len(token_clean):] if len(token) > len(token_clean) else ""
                for j, et in enumerate(expanded_tokens):
                    if j == len(expanded_tokens) - 1 and punct:
                        new_tokens.append(et + punct)
                    else:
                        new_tokens.append(et)
                replaced = True
                i += 1
                continue
            
            # Try to contract (check two-word combinations)
            if i + 1 < len(tokens):
                next_token_clean = tokens[i + 1].strip(".,!?;:")
                if next_token_clean:
                    contracted = contract_expanded(token_clean, next_token_clean)
                    if contracted:
                        # Preserve punctuation from second token
                        punct = tokens[i + 1][len(next_token_clean):] if len(tokens[i + 1]) > len(next_token_clean) else ""
                        new_tokens.append(contracted + punct)
                        replaced = True
                        i += 2  # Skip next token
                        continue
        
        # Stage 1: Synonym replacement
        if not replaced and len(token) > min_word_length and wordnet_pos:
            # Try domain-specific groups first
            group_synonym = get_synonym_from_group(token)
            if group_synonym:
                p_replace = {wordnet.ADJ: p_synonym_adj, wordnet.ADV: p_synonym_adv,
                            wordnet.NOUN: p_synonym_noun, wordnet.VERB: p_synonym_verb}.get(wordnet_pos, 0.45)
                if random.random() < p_replace:
                    new_tokens.append(preserve_capitalization(token, group_synonym))
                    replaced = True
            
            # Fallback to WordNet
            if not replaced:
                p_replace = {wordnet.ADJ: p_synonym_adj, wordnet.ADV: p_synonym_adv,
                            wordnet.NOUN: p_synonym_noun, wordnet.VERB: p_synonym_verb}.get(wordnet_pos, 0.45)
                if random.random() < p_replace:
                    wn_synonyms = get_wordnet_synonyms(token, pos)
                    if wn_synonyms:
                        new_tokens.append(preserve_capitalization(token, random.choice(wn_synonyms)))
                        replaced = True
        
        # Stage 2: Tense change for verbs (if not replaced by synonym)
        if not replaced and pos.startswith('V') and len(token) > 2:
            if random.random() < p_tense_change:
                tense_changed = change_verb_tense(token, pos)
                if tense_changed:
                    new_tokens.append(tense_changed)
                    replaced = True
        
        # Stage 3: Typo introduction (per example, not per sentence)
        if not replaced and len(token) > 3 and typos_count < max_typos_per_example:
            if random.random() < p_typo:
                word_typo, typo_ok = introduce_typo(token)
                if typo_ok:
                    new_tokens.append(word_typo)
                    typos_count += 1
                else:
                    new_tokens.append(token)
            else:
                new_tokens.append(token)
        elif not replaced:
            new_tokens.append(token)
        
        i += 1
    
    example["text"] = TreebankWordDetokenizer().detokenize(new_tokens)
    return example
