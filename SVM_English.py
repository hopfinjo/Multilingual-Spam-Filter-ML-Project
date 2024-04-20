import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("multilanguageemail.csv")

# Preprocess the English text data and convert it into numerical features using BoW with stopwords removal
vectorizer_english = CountVectorizer()
X_english = vectorizer_english.fit_transform(df['text'])  # Assuming 'text' contains the English email text
y_english = df['labels']

# Split the dataset into training and testing sets
X_train_english, X_test_english, y_train_english, y_test_english = train_test_split(X_english, y_english, test_size=0.2, random_state=42)

# Train the SVM classifier on English text
svm_classifier_english = SVC(kernel='linear')  # Linear kernel is often used for text classification
svm_classifier_english.fit(X_train_english, y_train_english)

# Evaluate the classifier on English text
y_pred_english = svm_classifier_english.predict(X_test_english)
accuracy_english = accuracy_score(y_test_english, y_pred_english)
print("Accuracy on English emails:", accuracy_english)

# You can also evaluate the classifier on German text using the existing SVM classifier trained on English text
# However, make sure to preprocess the German text using the same preprocessing steps applied to the English text
# and transform it using the English vectorizer

# Preprocess the German text data using the English vectorizer
X_german = vectorizer_english.transform(df['text_de'])  # Assuming 'text_de' contains the German email text

# Use the trained SVM classifier to predict the labels for the German emails
predicted_labels_german = svm_classifier_english.predict(X_german)

# Get the ground truth labels for the German emails
true_labels_german = df['labels']

# Calculate the accuracy on German emails
accuracy_german = accuracy_score(true_labels_german, predicted_labels_german)
print("Accuracy on English CL on German emails:", accuracy_german)
