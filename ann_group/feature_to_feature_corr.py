import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/F25/ECS170/ChessAI/ann_group/positions_with_features.csv')


features =["attack_balance","bishop_pair","bknrk_sqr_sum","center_attackers","connected_pawns","defense_balance",
    "doubled_pawns","half_open_king_files","king_ring_enemy_pressure",
    "material_bishop","material_knight","material_pawn","material_queen","material_rook","mobility_balance","mobility_safe_balance","outposts",
    "passed_pawns","pawn_shield","pawn_sqr_sum","pieces_occupying_center","queen_sqr_sum","threat_balance"
]

# map of feature labels, will be used in the bar plot
feature_labels = {"attack_balance": "abal","bishop_pair": "bbp","bknrk_sqr_sum": "bknrkss","center_attackers": "catk","connected_pawns": "cpwn","defense_balance": "dbal","doubled_pawns": "dpwn","half_open_king_files": "hogf","king_ring_enemy_pressure": "krep",
    "material_bishop": "mb", "material_knight": "mk", "material_pawn": "mp", "material_queen": "mq", "material_rook": "mr", "mobility_balance": "mba", "mobility_safe_balance": "msba", "outposts":"outp", "passed_pawns": "ppwn", "pawn_shield": "pwns",
    "pawn_sqr_sum": "pss","pieces_occupying_center": "pocc","queen_sqr_sum": "qss","threat_balance":"tbal",
}

corr_matrix = df[features].corr(method='pearson')

corr_matrix_labeled = corr_matrix.rename(columns=feature_labels, index=feature_labels)

plt.figure(figsize=(20, 10))
sns.heatmap(corr_matrix_labeled, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.3f')
plt.title('Feature-to-Feature Correlation Matrix', fontsize=12, fontweight='bold')

plt.xticks( fontsize=12, )
plt.yticks(fontsize=9, )
plt.tight_layout()
plt.show()


# features =["attack_balance","bishop_outposts","bishop_pair","bishop_sqr_sum","center_attackers","connected_pawns","defense_balance",
#     "doubled_pawns","half_open_king_files","isolated_pawns","king_ring_enemy_pressure","king_sqr_sum","knight_outposts","knight_sqr_sum",
#     "material_bishop","material_knight","material_pawn","material_queen","material_rook","mobility_balance","mobility_safe_balance",
#     "passed_pawns","pawn_shield","pawn_sqr_sum","pieces_occupying_center","queen_sqr_sum","rook_sqr_sum","threat_balance"
# ]
# feature_labels = {"attack_balance": "abal","bishop_outposts": "bop","bishop_pair": "bbp","bishop_sqr_sum": "bss","center_attackers": "catk","connected_pawns": "cpwn","defense_balance": "dbal","doubled_pawns": "dpwn","half_open_king_files": "hogf","isolated_pawns": "ipwn","king_ring_enemy_pressure": "krep","king_sqr_sum": "kiss","knight_outposts": "kop","knight_sqr_sum": "knss",
#     "material_bishop": "mb", "material_knight": "mk", "material_pawn": "mp", "material_queen": "mq", "material_rook": "mr", "mobility_balance": "mba", "mobility_safe_balance": "msba", "passed_pawns": "ppwn", "pawn_shield": "pwns",
#     "pawn_sqr_sum": "pss","pieces_occupying_center": "pocc","queen_sqr_sum": "qss","rook_sqr_sum": "rss","threat_balance":"tbal",
# }