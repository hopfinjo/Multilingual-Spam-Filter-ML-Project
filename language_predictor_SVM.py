
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Read the dataset
df = pd.read_csv("rearranged_dataset_lanugageprediction.csv")

# Split the dataset into features (text) and labels (language)
X = df['text']
y = df['language']

# Convert text data into numerical features using TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_features = vectorizer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

# Train the classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Evaluate the classifier
y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Predict language of new text samples
new_text_samples = [
    "This is a test sentence in English.",
    "मेरा नाम राहुल है।",
    "Guten Tag! Wie geht es Ihnen?",
    "Bonjour, comment ça va?"
]
new_text_features = vectorizer.transform(new_text_samples)
predicted_languages = svm_classifier.predict(new_text_features)
print("Predicted languages:", predicted_languages)
