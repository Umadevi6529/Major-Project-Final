import pandas as pd

# Read data from 'listresult.txt' and split into separate columns by spaces
data_list = []
with open('listresult.txt', 'r') as f:
    for line in f:
        data_list.append(line.strip().split())

# Convert the list of lists into a DataFrame
df = pd.DataFrame(data_list)

# Save the DataFrame to an Excel file
df.to_excel('output.xlsx', index=False, header=False)
