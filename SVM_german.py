import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("multilanguageemail.csv")

# Preprocess the German text data and convert it into numerical features using BoW with stopwords removal
vectorizer_german = CountVectorizer()  
X_german = vectorizer_german.fit_transform(df['text_de'])  # Assuming 'text_de' contains the German email text
y_german = df['labels']

# Split the dataset into training and testing sets
X_train_german, X_test_german, y_train_german, y_test_german = train_test_split(X_german, y_german, test_size=0.2, random_state=42)

# Train the SVM classifier on German text
svm_classifier_german = SVC(kernel='linear')  # Linear kernel is often used for text classification
svm_classifier_german.fit(X_train_german, y_train_german)

# Evaluate the classifier on German text
y_pred_german = svm_classifier_german.predict(X_test_german)
accuracy_german = accuracy_score(y_test_german, y_pred_german)
print("Accuracy on German emails:", accuracy_german)

# You can also evaluate the classifier on English text using the existing SVM classifier trained on English text
# However, make sure to preprocess the English text using the same preprocessing steps applied to the German text
# and transform it using the German vectorizer

# Preprocess the English text data using the German vectorizer
X_english = vectorizer_german.transform(df['text'])
# Assuming 'text' contains the English email text

# Use the trained SVM classifier to predict the labels for the English emails
predicted_labels_english = svm_classifier_german.predict(X_english)

# Get the ground truth labels for the English emails
true_labels_english = df['labels']

# Calculate the accuracy on English emails
accuracy_english = accuracy_score(true_labels_english, predicted_labels_english)
print("Accuracy on English emails using German classifier:", accuracy_english)
