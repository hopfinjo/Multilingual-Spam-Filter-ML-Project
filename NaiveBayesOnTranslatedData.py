import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Define a function to print accuracy and confusion matrix
def confusion_matrix_print(classifier, X_test, y_test):
    
    conf_matrix = confusion_matrix(y_test, classifier.predict(X_test))

    # Extract values from the confusion matrix
    true_positive = conf_matrix[1, 1]  # Correctly classified spam
    false_positive = conf_matrix[0, 1]  # Incorrectly classified spam
    true_negative = conf_matrix[0, 0]  # Correctly classified ham
    false_negative = conf_matrix[1, 0]  # Incorrectly classified ham

    # Print the statement
    print(f"Evaluation matrix: true positive spam: {true_positive}, false positive spam: {false_positive}, true pos ham {true_negative}, false pos ham {false_negative} \n")



# # Read the data from the CSV file

# data = pd.read_csv("translatedTextGerman.csv")

# # Separate English and German data
# english_data = data[data['text_en'].notna()]
# german_data = data[data['translated_de'].notna()]

# # Extract features (X) and labels (y) for English emails
# X_en = english_data['text_en']
# labels_en = english_data['labels']

# # Extract features (X) and labels (y) for German emails
# X_de = german_data['translated_de']
# labels_de = german_data['labels']


# Read the data from the CSV file
data = pd.read_csv("multilanguageemail.csv")

# Separate English and German data
english_data = data[data['text'].notna()]
german_data = data[data['text_de'].notna()]
french_data = data[data['text_fr'].notna()]

# Extract features (X) and labels (y) for English emails
X_en = english_data['text']
labels_en = english_data['labels']

# Extract features (X) and labels (y) for German emails
X_de = german_data['text_de']
labels_de = german_data['labels']

# Extract features (X) and labels (y) for French emails
X_fr = french_data['text_fr']
labels_fr = french_data['labels']

# Split the English data into training and testing sets
X_train_en, X_test_en, y_train_en, y_test_en = train_test_split(X_en, labels_en, test_size=0.4, random_state=44)

# Split the German data into training and testing sets
X_train_de, X_test_de, y_train_de, y_test_de = train_test_split(X_de, labels_de, test_size=0.4, random_state=44)

# Split the French data into training and testing sets
X_train_fr, X_test_fr, y_train_fr, y_test_fr = train_test_split(X_fr, labels_fr, test_size=0.4, random_state=44)


# Create a CountVectorizer to convert text into a matrix of token counts for English emails
vectorizer_en = CountVectorizer()
X_train_counts_en = vectorizer_en.fit_transform(X_train_en)
X_test_counts_en = vectorizer_en.transform(X_test_en)

# Create a CountVectorizer to convert text into a matrix of token counts for German emails
vectorizer_de = CountVectorizer()
X_train_counts_de = vectorizer_de.fit_transform(X_train_de)
X_test_counts_de = vectorizer_de.transform(X_test_de)

# Create a CountVectorizer to convert text into a matrix of token counts for French emails
vectorizer_fr = CountVectorizer()
X_train_counts_fr = vectorizer_fr.fit_transform(X_train_fr)
X_test_counts_fr = vectorizer_fr.transform(X_test_fr)


# Train a Multinomial Naive Bayes classifier for English emails
clf_en = MultinomialNB()
clf_en.fit(X_train_counts_en, y_train_en)

# Train a Multinomial Naive Bayes classifier for German emails
clf_de = MultinomialNB()
clf_de.fit(X_train_counts_de, y_train_de)

# Train a Multinomial Naive Bayes classifier for French emails
clf_fr = MultinomialNB()
clf_fr.fit(X_train_counts_fr, y_train_fr)

# ---------------------------------------------------------------------------------------

# Retrieve the log probabilities of features given each class
feature_log_probs_fr = clf_fr.feature_log_prob_

# The shape of feature_log_probs_en will be (n_classes, n_features),
# where n_classes is the number of target classes and n_features is the number of features

# If you want to associate each log probability with its corresponding feature,
# you can use the vocabulary provided by the CountVectorizer
feature_names_fr = vectorizer_fr.get_feature_names_out()

# Create a DataFrame to associate feature names with their log probabilities for each class
feature_importance_df_en = pd.DataFrame(feature_log_probs_fr, columns=feature_names_fr)


feature_probability_df = pd.DataFrame(feature_log_probs_fr, index=['Class 0', 'Class 1'], columns=feature_names_fr)

# Transpose the DataFrame to have words as column headers
feature_probability_df_transposed = feature_probability_df.transpose()

# Write the transposed DataFrame to a CSV file
feature_probability_df_transposed.to_csv('feature_probability.csv')


# print(X_test_counts_en[0].shape[1])
# print(X_test_counts_de[0].shape[1])
# Calculate accuracy for English emails

print("On translated dataset:")
accuracy_en_test = accuracy_score(y_test_en, clf_en.predict(X_test_counts_en))
print("Accuracy of English classifier on English test data:", accuracy_en_test)

confusion_matrix_print(classifier=clf_en, X_test=X_test_counts_en, y_test=y_test_en)

# Calculate accuracy for German emails using the English classifier
accuracy_de_using_en = accuracy_score(y_test_de, clf_de.predict(X_test_counts_de))
print("Accuracy of German classifier on German test data:", accuracy_de_using_en)
confusion_matrix_print(classifier=clf_de, X_test=X_test_counts_de, y_test=y_test_de)


# Transform German test data using the English CountVectorizer
X_test_counts_de_english_vectorizer = vectorizer_en.transform(X_test_de)

# Predict labels for German test data using the English-trained classifier
y_pred_de_en = clf_en.predict(X_test_counts_de_english_vectorizer)

# Calculate accuracy for German emails using the English-trained classifier
accuracy_englclassifier_ongerman = accuracy_score(y_test_de, y_pred_de_en)
print("Accuracy of English classifier on German test data:", accuracy_englclassifier_ongerman)
# dskfhaslkdfjlaskjd add print confusion matrix here

# Transform English test data using the German CountVectorizer
X_test_counts_en_de = vectorizer_de.transform(X_test_en)

# Predict labels for English test data using the German-trained classifier
y_pred_en_de = clf_de.predict(X_test_counts_en_de)

# Calculate accuracy for English emails using the German-trained classifier
accuracy_germancl_on_de = accuracy_score(y_test_en, y_pred_en_de)
print("Accuracy of German classifier on English test data:", accuracy_germancl_on_de)

# dskfhaslkdfjlaskjd add print confusion matrix here




accuracy_fr_test = accuracy_score(y_test_fr, clf_fr.predict(X_test_counts_fr))
print("Accuracy of French classifier on French test data:", accuracy_fr_test)

confusion_matrix_print(classifier=clf_fr, X_test=X_test_counts_fr, y_test=y_test_fr)





# Transform text_fr data using the English CountVectorizer
X_text_fr_counts_en = vectorizer_en.transform(english_data['text_fr'])

# Predict labels for text_fr data using the English-trained classifier
y_pred_text_fr_en = clf_en.predict(X_text_fr_counts_en)

# Calculate accuracy for text_fr data using the English-trained classifier
accuracy_text_fr_en = accuracy_score(english_data['labels'], y_pred_text_fr_en)
print("Accuracy of English classifier on text_fr data:", accuracy_text_fr_en)
confusion_matrix_print(classifier=clf_en, X_test=X_text_fr_counts_en, y_test=english_data['labels'])

# Transform text_fr data using the German CountVectorizer
X_text_fr_counts_de = vectorizer_de.transform(german_data['text_fr'])

# Predict labels for text_fr data using the German-trained classifier
y_pred_text_fr_de = clf_de.predict(X_text_fr_counts_de)

# Calculate accuracy for text_fr data using the German-trained classifier
accuracy_text_fr_de = accuracy_score(german_data['labels'], y_pred_text_fr_de)
print("Accuracy of German classifier on text_fr data:", accuracy_text_fr_de)
confusion_matrix_print(classifier=clf_de, X_test=X_text_fr_counts_de, y_test=german_data['labels'])

# Transform English text using the French CountVectorizer
X_text_en_counts_fr = vectorizer_fr.transform(english_data['text'])

# Predict labels for English text using the French-trained classifier
y_pred_text_en_fr = clf_fr.predict(X_text_en_counts_fr)

# Calculate accuracy for English text using the French-trained classifier
accuracy_text_en_fr = accuracy_score(english_data['labels'], y_pred_text_en_fr)
print("Accuracy of French classifier on English text data:", accuracy_text_en_fr)
confusion_matrix_print(classifier=clf_fr, X_test=X_text_en_counts_fr, y_test=english_data['labels'])

# Transform German text using the French CountVectorizer
X_text_de_counts_fr = vectorizer_fr.transform(german_data['text_de'])

# Predict labels for German text using the French-trained classifier
y_pred_text_de_fr = clf_fr.predict(X_text_de_counts_fr)

# Calculate accuracy for German text using the French-trained classifier
accuracy_text_de_fr = accuracy_score(german_data['labels'], y_pred_text_de_fr)
print("Accuracy of French classifier on German text data:", accuracy_text_de_fr)
confusion_matrix_print(classifier=clf_fr, X_test=X_text_de_counts_fr, y_test=german_data['labels'])

