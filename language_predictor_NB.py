import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import text

# Read the dataset
df = pd.read_csv("rearranged_dataset_lanugageprediction.csv")

# Split the dataset into features (text) and labels (language)
X = df['text']
y = df['language']


stop_words = text.ENGLISH_STOP_WORDS.union(["मेरा", "नाम", "राहुल", "है", "guten", "tag", "es", "ihnen", "bonjour", "ça", "va"])

# Convert text data into numerical features using TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words=stop_words)
X_features = vectorizer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.3, random_state=42)

# Train the Naive Bayes classifier
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train, y_train)

# Evaluate the classifier
y_pred = naive_bayes_classifier.predict(X_test)
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
predicted_languages = naive_bayes_classifier.predict(new_text_features)
print("Predicted languages:", predicted_languages)
