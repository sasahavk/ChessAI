
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv(
    'C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/F25/ECS170/ChessAI/ann_group/positions_with_features2.csv')

# Optional filters (keep if you want)
# df = df[df['move_num'] <= 40]
# df = df[df['target'].abs() > 1.0]

print(f"Total positions after filtering: {len(df):,}\n")
print("=" * 100)

# Feature list
features = [
    "attack_balance", "bishop_pair_black", "bishop_pair_white", "bknrk_sqr_sum",
    "center_attackers_black", "center_attackers_white", "connected_pawns",
    "defense_balance", "doubled_pawns", "half_open_king_files",
    "king_ring_enemy_pressure",
    "material_bishop_black", "material_bishop_white", "material_knight_black",
    "material_knight_white", "material_pawn_black", "material_pawn_white",
    "material_queen_black", "material_queen_white", "material_rook_black",
    "material_rook_white", "mobility_balance", "mobility_safe_balance",
    "outposts_black", "outposts_white", "passed_pawns", "pawn_shield",
    "pawn_sqr_sum", "pieces_occupying_center", "queen_sqr_sum", "threat_balance"
]

# Define your custom bins
bins = [0, 20, 40, 60, float('inf')]
labels = ["0–20", "20–40", "40–60", "60+"]

df['move_bin'] = pd.cut(df['move_num'], bins=bins, labels=labels,
                        include_lowest=True)

print("ZERO PERCENTAGE PER MOVE NUMBER BIN")
print("=" * 100)
print(
    f"{'Feature':<35} {'0–20':>10} {'20–40':>10} {'40–60':>10} {'60+':>10} {'Overall':>10}")
print("-" * 100)

for feat in features:
    row = []
    overall_zero = (df[feat] == 0).mean() * 100

    for bin_label in labels:
        subset = df[df['move_bin'] == bin_label]
        if len(subset) == 0:
            row.append("    -    ")
        else:
            zero_pct = (subset[feat] == 0).mean() * 100
            row.append(f"{zero_pct:7.1f}%")

    row.append(f"{overall_zero:7.1f}%")
    print(f"{feat:<35} " + " ".join(row))

print("\n" + "=" * 100)
print("POSITIONS PER BIN:")
print(df['move_bin'].value_counts().sort_index())
print("=" * 100)