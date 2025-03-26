import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

sentence = "I am learning NLP(Natural Language Processing)"
tokens = word_tokenize(sentence)


unigrams = list(ngrams(tokens, 1))
print('\n Unigrams:', unigrams )

bigrams = list(ngrams(tokens, 2))
print("\n Bigrams:", bigrams)

trigrams = list(ngrams(tokens, 3))
print('\n Triagrams:', trigrams)

print(bigrams)