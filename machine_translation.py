from translate import Translator

#create a translator object

translator = Translator(to_lang='ko') #spanish

#text to be translated

text = 'Hello, its a pleasure to meet you'

# Perform the translation
translation = translator.translate(text)

#print teh translated text
print("Translated text:", translation)