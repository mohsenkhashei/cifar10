import pandas as pd

# Read parameters from Excel file
params_df = pd.read_excel('parameters.xlsx')  # Adjust the file name as needed
params_row = params_df.iloc[-1]  # Use the last row

# Extract parameters
epochs = params_row['epochs']
batch_size = params_row['batch_size']
validation_split = params_row['validation_split']

print(params_row['optimizer']);