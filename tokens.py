import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

sample_text = 'I LOVE PROGRAMMING'
tokens = word_tokenize(sample_text.lower())

print('Tokens:', tokens)