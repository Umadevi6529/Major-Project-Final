import re

# Define a function to clean the input data
def clean_data(input_string):
    # Remove non-alphanumeric characters and extra spaces
    cleaned_string = re.sub(r'[^a-zA-Z0-9\s]', ' ', input_string)
    # Remove extra spaces
    cleaned_string = ' '.join(cleaned_string.split())
    return cleaned_string

# Read data from 'extracted.txt' and clean it
cleaned_data_list = []
with open('extracted.txt', 'r') as f:
    for line in f:
        cleaned_line = clean_data(line)
        if cleaned_line:  # Check if line is not empty after cleaning
            cleaned_data_list.append(cleaned_line)

# Write the cleaned data to 'listresult.txt'
with open('listresult.txt', 'w') as f:
    for line in cleaned_data_list:
        f.write(line + '\n')
