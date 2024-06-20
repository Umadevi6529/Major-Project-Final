import pandas as pd

# Assuming df is your DataFrame
# Example DataFrame
data = "listresult.txt"
df = pd.read_csv(data, sep=' ')

# Save DataFrame to a space-separated file
file_path = 'dataframe.csv'  # Specify the file path
df.to_csv(file_path, sep=' ', index=False)

print(f"DataFrame saved to {file_path} with space-separated values.")