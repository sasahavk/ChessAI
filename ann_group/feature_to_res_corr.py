import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/F25/ECS170/ChessAI/ann_group/positions_with_features.csv')

# features of the movies_features table/file, with target being average movie rating
# features =["attack_balance","bishop_pair","bknrk_sqr_sum","center_attackers","connected_pawns","defense_balance",
#     "doubled_pawns","half_open_king_files","king_ring_enemy_pressure",
#     "material_bishop","material_knight","material_pawn","material_queen","material_rook","mobility_balance","mobility_safe_balance", "outposts",
#     "passed_pawns","pawn_shield","pawn_sqr_sum","pieces_occupying_center","queen_sqr_sum","threat_balance"
# ]

features = ["attack_balance","bknrk_sqr_sum","center_attackers_black","center_attackers_white","connected_pawns",
            "defense_balance","doubled_pawns","half_open_king_files","king_ring_enemy_pressure",
            "material_bishop_black","material_bishop_white","material_knight_black","material_knight_white","material_pawn_black","material_pawn_white","material_queen_black","material_queen_white", "material_rook_black","material_rook_white",
            "mobility_balance","mobility_safe_balance","passed_pawns","pawn_shield",
            "pawn_sqr_sum","pieces_occupying_center","queen_sqr_sum","threat_balance"]
target = 'target'

print(len(features))

# map of feature labels, will be used in the bar plot
feature_labels = {
    "attack_balance": "abal",
    # "bishop_outposts": "bop",
    "bishop_pair": "bbp",
    "bknrk_sqr_sum": "bknrkss",
    # "bishop_sqr_sum": "bss",
    "center_attackers": "catk",
    "connected_pawns": "cpwn",
    "defense_balance": "dbal",
    "doubled_pawns": "dpwn",
    "half_open_king_files": "hogf",
    # "isolated_pawns": "ipwn",
    "king_ring_enemy_pressure": "krep",
    # "king_sqr_sum": "kiss",
    # "knight_outposts": "kop",
    # "knight_sqr_sum": "knss",
    "material_bishop": "mb",
    "material_knight": "mk",
    "material_pawn": "mp",
    "material_queen": "mq",
    "material_rook": "mr",
    "mobility_balance": "mba",
    "mobility_safe_balance": "msba",
    "outposts":"outp",
    "passed_pawns": "ppwn",
    "pawn_shield": "pwns",
    "pawn_sqr_sum": "pss",
    "pieces_occupying_center": "pocc",
    "queen_sqr_sum": "qss",
    # "rook_sqr_sum": "rss",
    "threat_balance":"tbal",
}
print(len(feature_labels))
correlations = {'Feature':[], 'Correlation':[]}

for f in features:
    correlations['Feature'].append(feature_labels[f])
    data = df[f]
    corr = data.corr(df[target], method='pearson')
    correlations['Correlation'].append(corr)

print(len(correlations['Feature']))
print(len(correlations['Correlation']))

corr_df = pd.DataFrame(correlations)
# round to 3 decimal places
corr_df['Correlation'] = corr_df['Correlation'].round(3)

for i in range(len(correlations["Feature"])):
    print(correlations['Feature'][i], correlations['Correlation'][i])

# create bar chart
plt.figure(figsize=(13, 6))
sns.barplot(x='Feature',  y='Correlation', data=corr_df, palette='coolwarm', hue='Correlation', dodge=False)

plt.ylabel('Pearson Correlation', fontweight='bold', fontsize=12)
plt.xlabel('Feature', fontweight='bold', fontsize=12)
plt.title('Correlation of Chess Features to Game Results', fontweight='bold', fontsize=12)
plt.ylim(-0.2, 0.2)
plt.axhline(0, color='black', linestyle='--', linewidth=0.4)
plt.xticks(fontsize=9)

for index, row in corr_df.iterrows():
    y_offset = 0.01 if row['Correlation'] >= 0 else -0.01
    plt.text(x=index, y=row['Correlation'] + y_offset, s=f"{row['Correlation']:.3f}",
             ha='center', va='bottom' if row['Correlation'] >= 0 else 'top', fontsize=9)

plt.legend([], [], frameon=False)

plt.tight_layout()
plt.show()

plt.close()

corr_df.to_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/F25/ECS170/ChessAI/ann_group/feature_corr2.csv',index=False)


# (board,turn,move_num,result,result_scaled,result_scaled_tempo,
#  attack_balance
#  bishop_pair_black
#  bishop_pair_white
#  bknrk_sqr_sum
#  center_attackers_black,
#  center_attackers_white
#  connected_pawns
#  defense_balance
#  doubled_pawns
#  half_open_king_files,
#  king_ring_enemy_pressure
#  material_bishop_black
#  material_bishop_white
#  material_knight_black
#  material_knight_white
#  material_pawn_black
#  material_pawn_white
#  material_queen_black
#  material_queen_white,
#  material_rook_black
#  material_rook_white
#  mobility_balance,
#  mobility_safe_balance
#  outposts_black
#  outposts_white
#  passed_pawns
#  pawn_shield
#  pawn_sqr_sum
#  pieces_occupying_center
#  queen_sqr_sum
#  target
#  threat_balance
