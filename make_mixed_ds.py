import pandas as pd

# Read the dataset
df = pd.read_csv("multilanguageemail.csv")

# Create lists to store combined text and labels
combined_text = []
labels = []

# Initialize counters for English, German, and French rows
en_index = 1
de_index = 2
fr_index = 0

# Iterate through the rows of the dataset
for index, row in df.iterrows():
    # Include English text
    if en_index % 3 == 0:
        combined_text.append(row['text'])
        labels.append(row['labels'])
    # Include German text
    elif de_index % 3 == 0:
        combined_text.append(row['text_de'])
        labels.append(row['labels'])
    # Include French text
    else:
        combined_text.append(row['text_fr'])
        labels.append(row['labels'])
    # Increment counters
    en_index += 1
    de_index += 1
    fr_index += 1

# Create a new dataframe with combined text and labels
combined_df = pd.DataFrame({'labels': labels, 'text_en_de_fr': combined_text})

# Write the new dataframe to a new CSV file
combined_df.to_csv("combined_multilanguage.csv", index=False)
