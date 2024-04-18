import pandas as pd
from openai import OpenAI
import os

def classify_email(text):
    os.environ["OPENAI_API_KEY"] = "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"

    client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a multilingual informal text message spam filter. Please classify this message as spam or ham. If you are unsure make it ham"},
            {"role": "user", "content": "Is this message spam or ham? Please classify it as spam or ham. Respond with either 'spam' or 'ham': " + text}
        ]
    )

    transl_text = completion.choices[0].message.content

    return transl_text

# Load the dataset
df = pd.read_csv("multilanguageemail.csv")

# Counters for statistics
correct_ham = 0
correct_spam = 0
incorrect = 0

# Iterate through each row in the dataset
for index, row in df.head(1000).iterrows():
    # Get the text and label from the dataset
    text = row['text']
    label = row['labels']
    
    # Classify the email
    classified_text = classify_email(text)
    print(classified_text)
    
    # Check if the classification matches the actual label
    if classified_text == label:
        if label == "ham":
            correct_ham += 1
        elif label == "spam":
            correct_spam += 1
    else:
        incorrect += 1

# Print statistics
print("Correctly classified ham:", correct_ham)
print("Correctly classified spam:", correct_spam)
print("Incorrectly classified:", incorrect)
