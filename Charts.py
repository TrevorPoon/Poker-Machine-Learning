import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

# Step 1: Read the CSV file into a Pandas DataFrame
# replace 'All_In_Simulation.csv' with the actual path to your CSV file
df = pd.read_csv('All_In_Simulation.csv', header=None, names=['Player1', 'Player2', 'P1_Wins', 'P2_Wins'])

# Step 2: Initialize the win matrix DataFrame
win_matrix = defaultdict(int)

# Count the wins for each hand pair
for index, row in df.iterrows():
    p1_hand = row['Player1']
    p2_hand = row['Player2']

    # Player 1 wins
    if row['P1_Wins'] == 2:
        win_matrix[p1_hand] += 1

    # Player 2 wins
    if row['P2_Wins'] == 2:
        win_matrix[p2_hand] += 1

# Step 3: Calculate the win rates

print(win_matrix)

win_matrix_df = pd.DataFrame(list(win_matrix.items()), columns=['Hand', 'Wins'])

# Get the total number of matchups for each hand pair
matchup_counts = len(df)

win_matrix_df['Win_Rate'] = win_matrix_df['Wins'] / matchup_counts


# Step 4: Create the heatmap

plt.figure(figsize=(12, 10))
sns.heatmap(win_matrix_df, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5)
plt.title('Poker Starting Hands Win Rate Matrix')
plt.xlabel('Player 2 Starting Hands')
plt.ylabel('Player 1 Starting Hands')
plt.show()