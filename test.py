import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Read the data from the CSV file
data = pd.read_csv("translatedTextGerman.csv")

# Separate English and German data
english_data = data[data['text_en'].notna()]
german_data = data[data['translated_de'].notna()]

# Extract features (X) and labels (y) for German emails
X_de = german_data['translated_de']
y_de = german_data['labels']

# Split the German data into training and testing sets
X_train_de, X_test_de, y_train_de, y_test_de = train_test_split(X_de, y_de, test_size=0.4, random_state=42)

# Create a CountVectorizer to convert text into a matrix of token counts for German emails
vectorizer_de = CountVectorizer()
X_train_counts_de = vectorizer_de.fit_transform(X_train_de)
X_test_counts_de = vectorizer_de.transform(X_test_de)

# Train a Multinomial Naive Bayes classifier for German emails
clf_de = MultinomialNB()
clf_de.fit(X_train_counts_de, y_train_de)

# Predict on the testing set for German emails
y_pred_de = clf_de.predict(X_test_counts_de)

# Calculate accuracy for German emails
accuracy_de = accuracy_score(y_test_de, y_pred_de)
print("Accuracy on German emails:", accuracy_de)

# Print classification report for German emails
print("Classification Report for German emails:")
print(classification_report(y_test_de, y_pred_de))
