import pandas as pd

# --------------------------------------------------------------
# 1. Your feature labels (keys only – the column names in CSV)
# --------------------------------------------------------------
feature_labels = {
    "attack_balance": "abal",
    "bishop_outposts": "bop",
    "bishop_pair": "bbp",
    "bishop_sqr_sum": "bss",
    "center_attackers": "catk",
    "connected_pawns": "cpwn",
    "defense_balance": "dbal",
    "doubled_pawns": "dpwn",
    "half_open_king_files": "hogf",
    "isolated_pawns": "ipwn",
    "king_ring_enemy_pressure": "krep",
    "king_sqr_sum": "kss",
    "knight_outposts": "kop",
    "knight_sqr_sum": "kss2",   # renamed to avoid duplicate
    "material_bishop": "mb",
    "material_knight": "mk",
    "material_pawn": "mp",
    "material_queen": "mq",
    "material_rook": "mr",
    "mobility_balance": "mbal",
    "mobility_safe_balance": "msbal",
    "passed_pawns": "ppwn",
    "pawn_shield": "pwns",
    "pawn_sqr_sum": "pss",
    "pieces_occupying_center": "pocc",
    "queen_sqr_sum": "qss",
    "rook_sqr_sum": "rss",
    "threat_balance": "tbal",
}

# Extract just the column names (keys)
FEATURE_COLUMNS = list(feature_labels.keys())


def analyze_feature_stats(
    csv_file='positions_with_features.csv',
    features=FEATURE_COLUMNS
):
    print(f"Loading CSV: {csv_file}")
    df = pd.read_csv(csv_file, usecols=lambda x: x in features or x == 'board')  # optional: keep board
    df = df[features]  # keep only feature columns

    print(f"\nAnalyzing {len(features)} features across {len(df):,} positions...\n")
    print("-" * 70)
    print(f"{'Feature':<30} {'Unique':>8} {'Zeros':>8} {'% Zeros':>8}")
    print("-" * 70)

    results = []
    for col in features:
        col_data = df[col]
        unique_count = col_data.nunique()
        zero_count = (col_data == 0).sum()
        zero_pct = zero_count / len(col_data) * 100 if len(col_data) > 0 else 0

        print(f"{col:<30} {unique_count:>8} {zero_count:>8} {zero_pct:>7.1f}%")
        results.append({
            'feature': col,
            'unique': unique_count,
            'zeros': zero_count,
            'zero_pct': zero_pct
        })

    print("-" * 70)
    return results

analyze_feature_stats()