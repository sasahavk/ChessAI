import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_exp = pd.read_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/F25/ECS170/ChessAI/positions_with_features.csv')

# features of the movies_features table/file, with target being average movie rating
features_exp =["attack_balance","bishop_outposts","bishop_pair","bishop_sqr_sum","center_attackers","connected_pawns","defense_balance",
    "doubled_pawns","half_open_king_files","isolated_pawns","king_ring_enemy_pressure","king_sqr_sum","knight_outposts","knight_sqr_sum",
    "material_bishop","material_knight","material_pawn","material_queen","material_rook","mobility_balance","mobility_safe_balance",
    "passed_pawns","pawn_shield","pawn_sqr_sum","pieces_occupying_center","queen_sqr_sum","rook_sqr_sum","threat_balance"
]
target = 'result'

# map of feature labels, will be used in the bar plot
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
    "knight_sqr_sum": "kss",
    "material_bishop": "mb",
    "material_knight": "mk",
    "material_pawn": "mp",
    "material_queen": "mq",
    "material_rook": "mr",
    "mobility_balance": "mba",
    "mobility_safe_balance": "msba",
    "passed_pawns": "ppwn",
    "pawn_shield": "pwns",
    "pawn_sqr_sum": "pss",
    "pieces_occupying_center": "pocc",
    "queen_sqr_sum": "qss",
    "rook_sqr_sum": "rss",
    "threat_balance":"tbal",
}

correlations = {'Feature':[], 'Correlation':[]}

for f in features_exp:
    correlations['Feature'].append(feature_labels[f])
    data = df_exp[f]
    corr = data.corr(df_exp[target], method='pearson')
    correlations['Correlation'].append(corr)

corr_df = pd.DataFrame(correlations)
# round to 3 decimal places
corr_df['Correlation'] = corr_df['Correlation'].round(3)

# create bar chart
plt.figure(figsize=(14, 6))
sns.barplot(x='Feature',  y='Correlation', data=corr_df, palette='coolwarm', hue='Correlation', dodge=False)

plt.ylabel('Pearson Correlation', fontweight='bold', fontsize=12)
plt.xlabel('Feature', fontweight='bold', fontsize=12)
plt.title('Correlation of Chess Features to Game Results', fontweight='bold', fontsize=12)
plt.ylim(-0.2, 0.6)
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.xticks(fontsize=12)

plt.legend([], [], frameon=False)

plt.tight_layout()
plt.show()

plt.close()

corr_df.to_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/F25/ECS170/ChessAI/feature_corr.csv',index=False)