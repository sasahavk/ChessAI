
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv(
    'C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/F25/ECS170/ChessAI/ann_group/positions_with_features_NEW.csv')

# Optional filters (keep if you want)
# df = df[df['move_num'] <= 40]
# df = df[df['target'].abs() > 3.0]

print(f"Total positions after filtering: {len(df):,}\n")
print("=" * 100)

# Feature list
features = [
    "attack_balance",
    "attack_black",
    "attack_white",
    "bishop_outposts_black",
    "bishop_outposts_white",
    "bishop_pair_white",
    "bishop_pair_black",
    "bishop_sqr_sum",
    "bishop_sqr_sum_black",
    "bishop_sqr_sum_white",
    "center_attackers_white",
    "center_attackers_black",
    "connected_pawns",
    "connected_pawns_black",
    "connected_pawns_white",
    "defense_balance",
    "defense_black",
    "defense_white",
    "doubled_pawns",
    "doubled_pawns_black",
    "doubled_pawns_white",
    "half_open_king_files",
    "half_open_king_files_black",
    "half_open_king_files_white",
    "isolated_pawns",
    "isolated_pawns_black",
    "isolated_pawns_white",
    "king_ring_enemy_pressure",
    "king_ring_enemy_pressure_black",
    "king_ring_enemy_pressure_white",
    "king_sqr_sum",
    "king_sqr_sum_black",
    "king_sqr_sum_white",
    "knight_outposts_black",
    "knight_outposts_white",
    "knight_sqr_sum",
    "knight_sqr_sum_black",
    "knight_sqr_sum_white",
    "material_bishop_white",
    "material_bishop_black",
    "material_knight_white",
    "material_knight_black",
    "material_pawn_white",
    "material_pawn_black",
    "material_queen_white",
    "material_queen_black",
    "material_rook_white",
    "material_rook_black",
    "mobility_balance",
    "mobility_black",
    "mobility_white",
    "mobility_safe_balance",
    "mobility_safe_black",
    "mobility_safe_white",
    "outposts_white",
    "outposts_black",
    "passed_pawns",
    "passed_pawns_black",
    "passed_pawns_white",
    "pawn_shield",
    "pawn_shield_black",
    "pawn_shield_white",
    "pawn_sqr_sum",
    "pawn_sqr_sum_black",
    "pawn_sqr_sum_white",
    "pieces_occupying_center",
    "queen_sqr_sum",
    "queen_sqr_sum_black",
    "queen_sqr_sum_white",
    "rook_sqr_sum",
    "rook_sqr_sum_black",
    "rook_sqr_sum_white",
    "target",
    "threat_balance",
    "threat_black",
    "threat_white"
]

# Define your feature columns (exclude 'target')
feature_cols = [col for col in features if col != 'target']

# Count how many features are exactly zero per row
zero_counts = (df[feature_cols] == 0).sum(axis=1)

# Get the distribution
dist = zero_counts.value_counts().sort_index()

# Print nice table
print("Number of positions with k features = 0:")
print("-" * 50)
for k, count in dist.items():
    print(f"{k:2d} features zero → {count:6,} positions ({count/len(df)*100:5.2f}%)")

print(f"\nMax number of zero features in one position: {zero_counts.max()}")
print(f"Average number of zero features per position: {zero_counts.mean():.2f}")

plt.figure(figsize=(12, 6))
ax = dist.plot(kind='bar', color='skyblue', edgecolor='black', width=0.8)

plt.title('Distribution: How Many Features Are Zero per Position\n'
          '(Filtered: |target| > 3.0)', fontsize=16, pad=20)
plt.xlabel('Number of Features Equal to Zero', fontsize=12)
plt.ylabel('Number of Positions', fontsize=12)
plt.xticks(rotation=0)

# Add count labels on top of bars
for i, (k, v) in enumerate(dist.items()):
    ax.text(i, v + len(df)*0.001, str(v), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()