import pandas as pd

# Load your file
df = pd.read_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/F25/ECS170/ChessAI/ann_group/positions_with_features_NEW.csv')


# Define the base piece names
pieces = ['bishop', 'rook', 'pawn', 'queen', 'knight', 'king']

# Black square-sum features
black_cols = [f"{piece}_sqr_sum_black" for piece in pieces]
print("Black columns:", black_cols)

# White square-sum features
white_cols = [f"{piece}_sqr_sum_white" for piece in pieces]
print("White columns:", white_cols)

# Compute the sums per row
df['sqr_sum_black'] = df[black_cols].sum(axis=1)
df['sqr_sum_white'] = df[white_cols].sum(axis=1)

# Optional: move the new columns near the original ones (for readability)
# Find where to insert them — after the last original sqr_sum column
insert_pos = df.columns.get_loc('rook_sqr_sum_white') + 1

# Reorder columns
cols = list(df.columns)
df = df[cols[:insert_pos] + ['sqr_sum_black', 'sqr_sum_white'] + cols[insert_pos:-2]]

# Save to new file
output_file = 'positions_with_features_NEW.csv'
df.to_csv(output_file, index=False)

print(f"\nDone! Added sqr_sum_black and sqr_sum_white")
print(f"Saved to: {output_file}")
print(f"Shape: {df.shape}")
print("\nFirst row example:")
print(df[['sqr_sum_black', 'sqr_sum_white']].head(1).to_string(index=False))