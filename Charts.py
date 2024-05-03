import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Read the Excel file into a Pandas DataFrame
df = pd.read_csv('All_In_Simulation.csv')

# Separating the DataFrame into two, one for each Player and their Wins
df1 = df[['Player1', 'P1_Wins']].rename(columns={'Player1': 'Cards', 'P1_Wins': 'Wins'})
df2 = df[['Player2', 'P2_Wins']].rename(columns={'Player2': 'Cards', 'P2_Wins': 'Wins'})

# Stack them on top of each other
df = pd.concat([df1, df2], ignore_index=True)

df = df.groupby('Cards')['Wins'].mean().reset_index()

# Renaming the column for clarity
df.rename(columns={'Wins': 'EV'}, inplace=True)

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
plt.title('Poker Preflop EV Matrix')
plt.xlabel('Suited')
plt.ylabel('Off-Suited')
plt.xticks(rotation=0)  # Ensures that the x-axis labels are horizontal for readability
# Move x-axis to top
plt.gca().xaxis.tick_top()
plt.gca().xaxis.set_label_position('top') 
plt.show()