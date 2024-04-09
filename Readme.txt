# maxis docu

datasets:
multilanguageemail: https://www.kaggle.com/datasets/rajnathpatel/multilingual-spam-data
"Context
This dataset is mainly used to test the zero-shot transfer for text classification using pretrained language models. The original text was in English and Machine Translated to Hindi, German and French."


emails_en_mostUsedWords:https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv
"About the Dataset
The csv file contains 5172 rows, each row for each email. There are 3002 columns. The first column indicates Email name.
 The name has been set with numbers and not recipients' name to protect privacy. 
The last column has the labels for prediction : 1 for spam, 0 for not spam. 
The remaining 3000 columns are the 3000 most common words in all the emails, after excluding the non-alphabetical characters/words. 
For each row, the count of each word(column) in that email(row) is stored in the respective cells. 
Thus, information regarding all 5172 emails are stored in a compact dataframe rather than as separate text files."




What am I gonna try:
First I will try to run simple classifier that are trained on one language also on other languages and see what happens 
(Is that even possible with the most common words. without translation? No.)

translate the dataset 

Try to run simple classifiers on multilingual data. Hence train it for some languages and see results.


After, I will implement language detection, and then run it through a classifier to get answers.

eventually with both datasets? 




possible other dataset that can be used
https://www.kaggle.com/datasets/mandygu/lingspam-dataset