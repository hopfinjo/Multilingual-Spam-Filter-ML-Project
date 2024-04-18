import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("multilanguageemail.csv")

# Preprocess the text data and convert it into numerical features using BoW with stopwords removal
vectorizer = CountVectorizer(stop_words='english')  # Remove English stopwords
X = vectorizer.fit_transform(df['text'])
y = df['labels']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Support Vector Machine classifier
svm_classifier = SVC(kernel='linear')  # Linear kernel is often used for text classification
svm_classifier.fit(X_train, y_train)

# Evaluate the classifier
y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


german_emails = df['text_de']  # Assuming 'text_de' contains the German email text
# Preprocess the German email text using the same preprocessing steps applied to the English emails

# Transform the preprocessed German email text into numerical features
X_german = vectorizer.transform(german_emails)

# Use the trained SVM classifier to predict the labels for the German emails
predicted_labels_german = svm_classifier.predict(X_german)

# Get the ground truth labels for the German emails
true_labels_german = df['labels']

# Calculate the accuracy
accuracy_german = accuracy_score(true_labels_german, predicted_labels_german)
print("Accuracy on German emails:", accuracy_german)