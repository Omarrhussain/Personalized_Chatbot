# Data Preprocessing Documentation

## Dataset: Persona-Chat
- Source: Kaggle
- Original size: X conversations
- Processed size: Y training pairs

## Preprocessing Steps:
1. **Data Cleaning**: Removed usernames, timestamps, non-textual data
2. **Tokenization**: Used NLTK word_tokenize
3. **Normalization**: Lowercasing, stopword removal, lemmatization
4. **Training Pairs**: Created input-response pairs from dialogues

## Rationale:
- Stopword removal: Reduce noise in training data
- Lemmatization: Maintain word meaning while reducing vocabulary size
- Pair creation: Enable supervised learning for chatbot responses