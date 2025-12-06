
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv(
    '/ann_group/positions_with_features_NEW.csv')

# Optional filters (keep if you want)
# df = df[df['move_num'] <= 40]
# df = df[df['target'].abs() > 3.0]


print(f"Total positions after filtering: {len(df):,}\n")
print("=" * 100)

# Feature list
features = [
    "attack_balance", #Y
    "attack_black",
    "attack_white",
    "bishop_outposts_black", # Z
    "bishop_outposts_white",
    "bishop_pair_white", # early
    "bishop_pair_black",
    "bishop_sqr_sum", #Y
    "bishop_sqr_sum_black",
    "bishop_sqr_sum_white",
    "center_attackers_white", #Y
    "center_attackers_black", #Y
    "connected_pawns", # Y
    "connected_pawns_black",
    "connected_pawns_white",
    "defense_balance", # Y
    "defense_black",
    "defense_white",
    "doubled_pawns", # zero
    "doubled_pawns_black",
    "doubled_pawns_white",
    "half_open_king_files", # late
    "half_open_king_files_black",
    "half_open_king_files_white",
    "isolated_pawns", # mide - late
    "isolated_pawns_black",
    "isolated_pawns_white",
    "king_ring_enemy_pressure", # Y
    "king_ring_enemy_pressure_black",
    "king_ring_enemy_pressure_white",
    "king_sqr_sum",
    "king_sqr_sum_black", # Y
    "king_sqr_sum_white", # Y
    "knight_outposts_black", # zero
    "knight_outposts_white", # zero
    "knight_sqr_sum", # early - mid
    "knight_sqr_sum_black",
    "knight_sqr_sum_white",
    "material_bishop_white", # early - mid
    "material_bishop_black", # early - mid
    "material_knight_white", # early - mid
    "material_knight_black", # early - mid
    "material_pawn_white", # Y
    "material_pawn_black", # Y
    "material_queen_white", # early - mid
    "material_queen_black", # early - mid
    "material_rook_white", # early - mid
    "material_rook_black", # early - mid
    "mobility_balance", # Y
    "mobility_black",
    "mobility_white",
    "mobility_safe_balance", # Y
    "mobility_safe_black",
    "mobility_safe_white",
    "outposts_white", # zero
    "outposts_black", # zero
    "passed_pawns", # mid - late
    "passed_pawns_black",
    "passed_pawns_white",
    "pawn_shield", # Y
    "pawn_shield_black",
    "pawn_shield_white",
    "pawn_sqr_sum", # Y
    "pawn_sqr_sum_black",
    "pawn_sqr_sum_white",
    "pieces_occupying_center", # Y
    "queen_sqr_sum", # zero
    "queen_sqr_sum_black",
    "queen_sqr_sum_white",
    "rook_sqr_sum", # zero
    "rook_sqr_sum_black",
    "rook_sqr_sum_white",
    "sqr_sum_black",
    "sqr_sum_white",
    "target",
    "threat_balance",
    "threat_black", # Y
    "threat_white" # Y
]

df = df[(df[features] == 0).sum(axis=1) <= 40]
df = df[df['target'].abs() > 1.0]


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
