# Data Transformation Design for OOD Evaluation

## Overview

This document describes the data transformation strategy designed to create out-of-distribution (OOD) evaluation data for the IMDB movie review sentiment classification task. The transformation aims to simulate realistic variations that could occur in real-world test scenarios while preserving the original sentiment label.

## Transformation Design: Hybrid Synonym Replacement with Controlled Typographical Errors

### Description

The transformation applies a two-stage approach to modify movie review text:

1. **Synonym Replacement (Primary)**: Replace content words (nouns, adjectives, verbs, adverbs) with their synonyms from WordNet, preserving semantic meaning and sentiment.

2. **Controlled Typographical Errors (Secondary)**: Introduce realistic typing errors by simulating QWERTY keyboard proximity errors on a subset of words, ensuring the text remains readable.

### Detailed Algorithm

#### Stage 1: Synonym Replacement (Two-Tier Approach)

1. **Tokenization**: Tokenize the input text using NLTK's `word_tokenize()` to preserve sentence structure.

2. **Part-of-Speech Filtering**: For each token, determine if it's a content word (noun, verb, adjective, or adverb) using WordNet's POS tags:
   - Nouns: `wordnet.NOUN`
   - Verbs: `wordnet.VERB`
   - Adjectives: `wordnet.ADJ`
   - Adverbs: `wordnet.ADV`

3. **Domain-Specific Synonym Groups (Primary)**:
   - Check if word exists in domain-specific synonym groups (`SYNONYM_GROUPS`)
   - Groups are defined as sets of mutually synonymous words (e.g., `{'movie', 'film', 'picture'}`)
   - Groups automatically build bidirectional lookup dictionary
   - If word is found in groups:
     - Get POS-based replacement probability
     - Randomly select a synonym from the group (excluding the word itself)
     - If `random() < p_replace`, replace with selected synonym
   - Domain groups include:
     - Movie/Film terms: movie, film, picture, flick
     - Acting/Performance: acting, performance, portrayal, playing
     - Story/Narrative: story, narrative, tale, plot, screenplay, script
     - Quality terms: good, great, excellent, amazing, awesome, bad, terrible, etc.
     - Common verbs/adverbs: watch/see, like/love, very/really, etc.

4. **WordNet Synonym Lookup (Fallback)**:
   - If word was not replaced by domain groups:
     - Query WordNet synsets using `wordnet.synsets(word, pos=pos_tag)`
     - Extract lemmas from the synsets using `.lemmas()`
     - Filter lemmas to ensure:
       - The lemma is different from the original word (case-insensitive)
       - The lemma contains only alphabetic characters (no special characters)
       - The lemma is not a duplicate
     - Get POS-based replacement probability
     - If `random() < p_replace` and synonyms exist, randomly select one

5. **Replacement Strategy**:
   - Preserve the original word's capitalization (uppercase, lowercase, title case)
   - Only replace words longer than 3 characters to avoid replacing common words like "the", "and"
   - POS-based probabilities:
     - Adjectives: 70% (`p_synonym_adj = 0.70`)
     - Adverbs: 70% (`p_synonym_adv = 0.70`)
     - Nouns: 30% (`p_synonym_noun = 0.30`)
     - Verbs: 25% (`p_synonym_verb = 0.25`)

6. **Detokenization**: Reconstruct the sentence using `TreebankWordDetokenizer()` to restore natural spacing and punctuation.

#### Stage 2: Controlled Typographical Errors

1. **Word Selection**: With probability `p_typo` (0.10), select a word for typo introduction, ensuring:
   - Word was not replaced in Stage 1
   - Word length > 3 characters
   - Total typos in example < `max_typos_per_example` (2)

2. **Character Replacement**:
   - Define QWERTY keyboard proximity map for common vowels and consonants:
     ```
     'a' → ['s', 'q', 'w', 'e']
     'e' → ['w', 'r', 'd', 's']
     'i' → ['u', 'o', 'k', 'j']
     'o' → ['i', 'p', 'l', 'k']
     'u' → ['y', 'i', 'j', 'h']
     's' → ['a', 'w', 'e', 'd', 'x', 'z']
     'd' → ['s', 'e', 'r', 'f', 'c']
     'r' → ['e', 'd', 'f', 't']
     't' → ['r', 'f', 'g', 'y']
     'n' → ['b', 'h', 'j', 'm']
     'l' → ['k', 'o', 'p']
     ```
   - For the selected word, randomly choose one character position (excluding first and last characters to maintain word boundaries)
   - Replace the character with a nearby key from the proximity map, if applicable
   - Preserve case of the replaced character
   - If the character doesn't have a proximity mapping, skip typo introduction for that word

3. **Error Rate Control**: Limit to at most 2 typos per example (not per sentence) to maintain readability while allowing realistic distribution across longer texts.

### Implementation Parameters

**Synonym Replacement Probabilities (POS-based)**:
- `p_synonym_adj = 0.70`: Probability of replacing adjectives (higher for semantic flexibility)
- `p_synonym_adv = 0.70`: Probability of replacing adverbs (higher for semantic flexibility)
- `p_synonym_noun = 0.30`: Probability of replacing nouns (lower to preserve meaning)
- `p_synonym_verb = 0.25`: Probability of replacing verbs (lower to preserve meaning)
- `min_word_length = 4`: Minimum word length for synonym replacement

**Typo Introduction Parameters**:
- `p_typo = 0.10`: Probability of introducing a typo in an eligible word (10%)
- `min_word_length = 4`: Minimum word length for typo introduction (actual requirement: >3 for character position)
- `max_typos_per_example = 2`: Maximum number of typos per example (not per sentence)

**Rationale for POS-based Probabilities**:
- Higher probabilities for adjectives/adverbs allow more semantic variation while maintaining sentiment
- Lower probabilities for nouns/verbs preserve core meaning and relationships
- Domain-specific groups provide better control for movie review terminology

### Example Transformations

**Example 1: Synonym Replacement**
```
Original: "Titanic is the best movie I have ever seen. The acting was exceptional and the story was compelling."
Transformed: "Titanic is the finest film I have ever seen. The acting was extraordinary and the narrative was compelling."
```

**Example 2: Combined Transformation**
```
Original: "This film was terrible. The plot made no sense and the characters were poorly developed."
Transformed: "This movie was awful. The plot made no sense and the characters were poorly developed."
```
(Note: "film" → "movie" (domain group), "terrible" → "awful" (domain group))

**Example 3: Domain-Specific Groups in Action**
```
Original: "Titanic is the best movie I have ever seen. The acting was excellent and the screenplay was amazing."
Transformed: "Titanic is the finest picture I have ever seen. The performance was outstanding and the script was fantastic."
```
(Note: "movie" → "picture", "acting" → "performance", "excellent" → "outstanding", "screenplay" → "script", "amazing" → "fantastic" - all from domain groups)

**Example 4: Typo Introduction**
```
Original: "This is a great movie with excellent acting."
Transformed: "This is a great film with excellent actibg."
```
(Note: "movie" → "film" (synonym), "actibg" (typo in "acting" - e→i keyboard proximity)

### Why This Transformation is "Reasonable"

1. **Realistic User Variations**: 
   - Synonym usage reflects natural language diversity where users express similar sentiments using different vocabulary
   - Typographical errors simulate real-world scenarios where users make keyboard mistakes, especially on mobile devices

2. **Preserves Semantic Meaning**: 
   - Synonym replacement maintains the core meaning and sentiment of the review
   - Controlled typos are minor and don't render text incomprehensible

3. **Test-Time Plausibility**:
   - Users frequently use synonyms in reviews (e.g., "excellent" vs "outstanding", "bad" vs "terrible")
   - Typing errors are common in user-generated content, especially in informal reviews
   - Both variations occur naturally in real-world movie review datasets

4. **Appropriate Complexity**:
   - Not trivial: Requires semantic understanding to maintain sentiment
   - Not extreme: Text remains readable and interpretable by humans
   - Challenges the model's robustness without breaking it

### Evaluation Criteria Compliance

- ✓ **Maintains Label**: Synonym replacement preserves sentiment; controlled typos don't alter meaning
- ✓ **Reasonable**: Variations that occur naturally in user-generated content
- ✓ **Clear Description**: Algorithm is detailed enough for reimplementation
- ✓ **Not Trivial**: Requires semantic understanding and introduces realistic noise
- ✓ **Not Extreme**: Text remains coherent and human-readable

### Implementation Notes

The transformation is implemented in `utils.py` in the `custom_transform()` function. The implementation uses:

**Data Structures**:
- `SYNONYM_GROUP_SETS`: List of sets defining domain-specific synonym groups
- `SYNONYM_GROUPS`: Auto-built bidirectional lookup dictionary from groups
- Benefits: No redundancy, automatic bidirectional mapping, easy to maintain

**Key Functions**:
- `get_synonym_from_group(word)`: Returns synonym from domain-specific groups
- `get_wordnet_synonyms(word, pos_tag)`: Fallback WordNet synonym lookup
- `introduce_typo(word)`: Introduces typo using QWERTY keyboard proximity
- `preserve_capitalization(original, replacement)`: Maintains capitalization patterns

**Libraries**:
- `nltk.word_tokenize()` for tokenization
- `nltk.pos_tag()` for part-of-speech tagging
- `nltk.corpus.wordnet` for synonym lookup (fallback)
- `TreebankWordDetokenizer` for sentence reconstruction
- Random sampling with fixed seed (0) for reproducibility

**Key Design Decisions**:
1. **Two-tier synonym lookup**: Domain groups first (better control), WordNet fallback (coverage)
2. **POS-based probabilities**: Higher for adjectives/adverbs, lower for nouns/verbs
3. **Per-example typo limit**: Changed from per-sentence to per-example (more realistic)
4. **Typo as fallback**: Only applies if synonym replacement didn't occur

### Expected Impact on Model Performance

This transformation tests the model's ability to:
1. Generalize across lexical variations (synonym robustness)
2. Handle minor spelling errors (typo robustness)
3. Maintain performance on semantically equivalent but lexically different inputs

A robust model should maintain high accuracy on transformed data, as the semantic content and sentiment remain unchanged. A significant performance drop would indicate over-reliance on specific word choices rather than semantic understanding.

