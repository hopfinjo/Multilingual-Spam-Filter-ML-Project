import pandas as pd
from openai import OpenAI
import os

def translateMe(text):
    os.environ["OPENAI_API_KEY"] = "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"

    client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a interpreter/translater expert that translates email texts into other languages. You keep their original form and translate them"},
            {"role": "user", "content": "Translate this to german" + text}
        ]
    )

    transl_text = completion.choices[0].message.content

    return transl_text

def translate_csv(input_file, output_file):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(input_file)

    # Open the output file for writing
    with open(output_file, 'w', encoding='utf-8') as output:
        # Write the header to the output file
        output.write('labels,text_en,translated_de\n')

        # Iterate through each row in the DataFrame
        for index, row in df.iterrows():
            print(index)
            # Apply the translation function to each text entry in the 'text_en' column
            translated_text = translateMe(row["text_en"])

            # Write the translated row to the output file
            output.write(f"{row['labels']},\"{row['text_en']}\",\"{translated_text}\"\n")

            # Flush the buffer to ensure the data is written immediately
            output.flush()

# Replace 'input_file.csv' with the path to your input CSV file
input_file = 'email_EnOnly.csv'

# Call the function to translate the CSV file
translate_csv(input_file, 'translatedTextGerman.csv')