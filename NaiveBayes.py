import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Read the data from the CSV file
data = pd.read_csv("translatedTextGerman.csv")

# Separate English and German data
english_data = data[data['text_en'].notna()]
german_data = data[data['translated_de'].notna()]

# Extract features (X) and labels (y) for English emails
X_en = english_data['text_en']
y_en = english_data['labels']

# Split the English data into training and testing sets
X_train_en, X_test_en, y_train_en, y_test_en = train_test_split(X_en, y_en, test_size=0.1, random_state=42)

# Create a CountVectorizer to convert text into a matrix of token counts for English emails
vectorizer_en = CountVectorizer()
X_train_counts_en = vectorizer_en.fit_transform(X_train_en)
X_test_counts_en = vectorizer_en.transform(X_test_en)

# Train a Multinomial Naive Bayes classifier for English emails
clf_en = MultinomialNB()
clf_en.fit(X_train_counts_en, y_train_en)

# Calculate accuracy for English emails
accuracy_en_train = accuracy_score(y_train_en, clf_en.predict(X_train_counts_en))
accuracy_en_test = accuracy_score(y_test_en, clf_en.predict(X_test_counts_en))
print("Accuracy on English training data:", accuracy_en_train)
print("Accuracy on English testing data:", accuracy_en_test)

# Extract features (X) and labels (y) for German emails
X_de = german_data['translated_de']
y_de = german_data['labels']

# Transform German text using the CountVectorizer trained on English data
X_de_counts = vectorizer_en.transform(X_de)

# Predict labels using the trained English classifier for German emails
y_pred_de = clf_en.predict(X_de_counts)

# Calculate accuracy for German emails
accuracy_de = accuracy_score(y_de, y_pred_de)
print("Accuracy on German emails using English classifier:", accuracy_de)
