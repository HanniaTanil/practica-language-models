# simple machine learning script for text classification using Naive Bayes
# specifically designed to categorize text as either "tech" or "non-tech"

from sklearn.feature_extraction.text import CountVectorizer # Converts text into a matrix of token counts 
from sklearn.naive_bayes import MultinomialNB # Naive Bayes classifier for text classification
from sklearn.model_selection import train_test_split # Splits the data into training and testing sets  
from sklearn.metrics import accuracy_score # Calculates the accuracy of the model's predictions

# Prepare the data
# We need to crate a dataset of texts and their correspoding labels. Each text is either related to technology ("tech") or not ("non-tech")

texts = ['I love programming', 'Python is amazing', 'I enjoy machine learning',
        'The weather is nice today', 'I like algo.', 'Machine learning is fascinating',
        'Natural Language Processing is a part of AI']

labels = ['tech', 'tech', 'tech', 'non-tech', 'tech', 'tech', 'tech']

# Convert text to numerical data
# We will use CountVectorizer to convert our text data into a matrix of tokens
# and fit_transform() learns the vocabulary and transforms the text into a matrix of numbers

vectorizer = CountVectorizer()
x = vectorizer.fit_transform(texts)

# Split the data into training
# The train_test_split function splits the data into training (80%) and testing (20%) sets
# random_state=42 ensures the reproducibility of the split

x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=42)

# Train the Naive Bayes Classifier
# Use the MultinomialNB clasifier to train the model on our training data

model = MultinomialNB()
model.fit(x_train, y_train)

#Make predictions on the Test Set
# Use the trained model to predict the labels for the test set

y_pred = model.predict(x_test)

# Evaluate the models accuracy
# Calculate and print te accuracy of our model by comparing the predicted labels with the actual labels in the test set

accuracy =  accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)