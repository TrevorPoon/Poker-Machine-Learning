import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Read the Excel file into a Pandas DataFrame
df = pd.read_excel('All_In_Simulation.xlsx')

# Step 2: Create a hand matrix (13x13 representing all card combinations)
# Initialize an empty 13x13 matrix filled with NaN
hand_matrix = pd.DataFrame(np.nan, index=list('AKQJT98765432'), columns=list('AKQJT98765432'))

# Fill the matrix with EV values from the DataFrame
for index, row in df.iterrows():
    # Cards are in format like 'AAO' or 'AKS'
    # Extract the first character for the row index, second for the column index
    row_index = row['Cards'][0]  # The first card
    col_index = row['Cards'][1]  # The second card

    # If 'O' for off-suit, place the EV in both (row, col) and (col, row)
    if 'O' in row['Cards']:
        hand_matrix.at[col_index, row_index] = row['EV']
    # If 'S' for suited, place the EV only in (row, col) since suited cards must be in order
    elif 'S' in row['Cards']:
        hand_matrix.at[row_index, col_index] = row['EV']

# Replace NaN with zero or any appropriate value for combinations that don't exist
hand_matrix.fillna(0, inplace=True)

# Step 3: Create the heatmap
plt.figure(figsize=(10, 8))  # Size for readability
sns.heatmap(hand_matrix, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5)
plt.title('Poker Starting Hands EV Matrix')
plt.xlabel('Suited')
plt.ylabel('Off')
plt.show()