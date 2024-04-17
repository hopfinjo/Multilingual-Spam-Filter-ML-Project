import pandas as pd
import matplotlib.pyplot as plt





# Read the CSV file
df = pd.read_csv("feature_probability_abs_de_kaggledataset.csv")

# # Apply absolute function to each value in the DataFrame
# df['Class 0'] = df['Class 0'].abs()
# df['Class 1'] = df['Class 1'].abs()

# df.to_csv("feature_probability_abs.csv", index=False)


# Calculate the difference between Class 0 and Class 1 absolute values
df['Difference'] = df['Class 0'] - df['Class 1']

# Sort the DataFrame based on the difference in descending order
sorted_df = df.sort_values(by='Difference', ascending=False)

# Get the top 20 words with the highest difference
top_20_words = sorted_df.head(20)

# Plot the differences as a column chart
plt.figure(figsize=(10, 6))
plt.barh(top_20_words['words'], top_20_words['Difference'], color='skyblue')
plt.xlabel('Difference')
plt.ylabel('Feature Weight')
plt.title('Top 20 Words with highest weight to be spam German cf KAGGLE DS')
plt.gca().invert_yaxis()
plt.show()
