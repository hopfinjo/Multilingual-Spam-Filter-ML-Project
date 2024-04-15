import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Read the data from the CSV file
data = pd.read_csv("translatedTextGerman.csv")

# Extract features (X) and labels (y)
X = data['text']
y = data['labels']

# Split the data into training and testing sets for English emails
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Create a CountVectorizer to convert text into a matrix of token counts for English emails
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train a Multinomial Naive Bayes classifier for English emails
clf = MultinomialNB()
clf.fit(X_train_counts, y_train)

# Predict on the testing set for English emails
y_pred = clf.predict(X_test_counts)

# Calculate accuracy for English emails
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on English emails:", accuracy)

# Print classification report for English emails
print("Classification Report for English emails:")
print(classification_report(y_test, y_pred))

# Extract German text (text_de)
X_de = data['translated_de']

# Transform German text using the same CountVectorizer
X_de_counts = vectorizer.transform(X_de)

# Predict labels using the trained classifier for German emails
y_pred_de = clf.predict(X_de_counts)

# Extract true labels for the German emails
y_true_de = data['labels']

# Calculate accuracy for German emails
accuracy_de = accuracy_score(y_true_de, y_pred_de)
print("Accuracy on German emails:", accuracy_de)

# Print classification report for German emails
print("Classification Report for German emails:")
print(classification_report(y_true_de, y_pred_de))
