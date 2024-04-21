import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Read the dataset
df = pd.read_csv("combined_multilanguage_defren.csv")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text_en_de_fr'], df['labels'], test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Support Vector Machine classifier
classifier = SVC(kernel='linear')  # You can choose different kernels such as 'linear', 'rbf', 'poly', etc.
classifier.fit(X_train_vec, y_train)

# Make predictions
y_pred = classifier.predict(X_test_vec)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
