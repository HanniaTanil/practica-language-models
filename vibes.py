from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from textblob import TextBlob

# Expanded sample movie reviews
reviews = [
    "This movie was awful",
    "Worst movie ever", 
    "Amazing story line and great acting!",
    "The plot was cringe.",
    "Not recommended.",
    "Absolutely terrible film",
    "Incredible performances!",
    "Boring and predictable",
    "A masterpiece of cinema",
    "Complete waste of time",
    "This movie was awful",
    "Worst movie ever", 
    "Amazing story line and great acting!",
    "The plot was cringe.",
    "Not recommended.",
    "Absolutely terrible film",
    "Incredible performances!",
    "Boring and predictable",
    "A masterpiece of cinema",
    "Complete waste of time"
]

# Labels for the reviews
labels = [
    "negative", "negative", "positive", 
    "negative", "negative", "negative", 
    "positive", "negative", "positive", 
    "negative", "negative", "negative", 
    "positive", "negative", "negative", 
    "negative", "positive", "negative", 
    "positive", "negative"
]

# Vectorize the data 
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(reviews)

# Split the data with a larger test size
x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.3, random_state=42)

# Create a Naive Bayes classifier
model = MultinomialNB()

# Train the model
model.fit(x_train, y_train)

# Use the trained model to make predictions on the test data
y_pred = model.predict(x_test)

# Calculate the accuracy of the model 
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', accuracy)

# Interpret the results 
if accuracy > 0.5:
    print('Good vibes. Book the ticket!')
else:
    print('Needs more work!')