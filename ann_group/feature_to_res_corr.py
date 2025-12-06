import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/ann_group/positions_with_features_NEW.csv')

# features of the movies_features table/file, with target being average movie rating
# features =["attack_balance","bishop_pair","bknrk_sqr_sum","center_attackers","connected_pawns","defense_balance",
#     "doubled_pawns","half_open_king_files","king_ring_enemy_pressure",
#     "material_bishop","material_knight","material_pawn","material_queen","material_rook","mobility_balance","mobility_safe_balance", "outposts",
#     "passed_pawns","pawn_shield","pawn_sqr_sum","pieces_occupying_center","queen_sqr_sum","threat_balance"
# ]

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
    "threat_balance",
    "threat_black", # Y
    "threat_white" # Y
]
target = 'target'

print(len(features))

# map of feature labels, will be used in the bar plot
feature_labels = {
    "attack_balance":"attb", #Y
    "attack_black":"attB",
    "attack_white":"attW",
    "bishop_outposts_black":"bopB", # Z
    "bishop_outposts_white":"bopW",
    "bishop_pair_white":"bpW", # early
    "bishop_pair_black":"bpB",
    "bishop_sqr_sum":"bss", #Y
    "bishop_sqr_sum_black":"bssB",
    "bishop_sqr_sum_white":"bssW",
    "center_attackers_white":"cattW", #Y
    "center_attackers_black":"cattB", #Y
    "connected_pawns":"cpwn", # Y
    "connected_pawns_black":"cpwnB",
    "connected_pawns_white":"cpwnW",
    "defense_balance":"defb", # Y
    "defense_black":"defB",
    "defense_white":"defW",
    "doubled_pawns":"dbpwn", # zero
    "doubled_pawns_black":"dbpwnB",
    "doubled_pawns_white":"dbpwnW",
    "half_open_king_files":"hokf", # late
    "half_open_king_files_black":"hokfB",
    "half_open_king_files_white":"hokfW",
    "isolated_pawns":"ipwn", # mide - late
    "isolated_pawns_black":"ipwnB",
    "isolated_pawns_white":"ipwnW",
    "king_ring_enemy_pressure":"krep", # Y
    "king_ring_enemy_pressure_black":"krepB",
    "king_ring_enemy_pressure_white":"krepW",
    "king_sqr_sum":"kiss",
    "king_sqr_sum_black":"kissB", # Y
    "king_sqr_sum_white":"kissW", # Y
    "knight_outposts_black":"kopB", # zero
    "knight_outposts_white":"kopW", # zero
    "knight_sqr_sum":"knss", # early - mid
    "knight_sqr_sum_black":"knssB",
    "knight_sqr_sum_white":"knssW",
    "material_bishop_white":"mbW", # early - mid
    "material_bishop_black":"mbB", # early - mid
    "material_knight_white":"mknW", # early - mid
    "material_knight_black":"mknB", # early - mid
    "material_pawn_white":"mpwnW", # Y
    "material_pawn_black":"mpwnB", # Y
    "material_queen_white":"mqW", # early - mid
    "material_queen_black":"mqB", # early - mid
    "material_rook_white":"mrW", # early - mid
    "material_rook_black":"mrB", # early - mid
    "mobility_balance":"mob", # Y
    "mobility_black":"mobB",
    "mobility_white":"mobW",
    "mobility_safe_balance":"mobs", # Y
    "mobility_safe_black":"mobsB",
    "mobility_safe_white":"mobsW",
    "outposts_white":"opW", # zero
    "outposts_black":"opB", # zero
    "passed_pawns":"ppwn", # mid - late
    "passed_pawns_black":"ppwnB",
    "passed_pawns_white":"ppwnW",
    "pawn_shield":"psh", # Y
    "pawn_shield_black":"pshB",
    "pawn_shield_white":"pshW",
    "pawn_sqr_sum":"pwnss", # Y
    "pawn_sqr_sum_black":"pwnssB",
    "pawn_sqr_sum_white":"pwnssW",
    "pieces_occupying_center":"poc", # Y
    "queen_sqr_sum":"qss", # zero
    "queen_sqr_sum_black":"qssB",
    "queen_sqr_sum_white":"qssW",
    "rook_sqr_sum":"rss", # zero
    "rook_sqr_sum_black":"rssB",
    "rook_sqr_sum_white":"rssW",
    "sqr_sum_black":"ssB",
    "sqr_sum_white":"ssW",
    "threat_balance":"thrt",
    "threat_black":"thrtB", # Y
    "threat_white":"thrtW" # Y
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

# # create bar chart
# plt.figure(figsize=(13, 6))
# sns.barplot(x='Feature',  y='Correlation', data=corr_df, palette='coolwarm', hue='Correlation', dodge=False)
#
# plt.ylabel('Pearson Correlation', fontweight='bold', fontsize=10)
# plt.xlabel('Feature', fontweight='bold', fontsize=10)
# plt.title('Correlation of Chess Features to Game Results', fontweight='bold', fontsize=10)
# plt.ylim(-0.2, 0.2)
# plt.axhline(0, color='black', linestyle='--', linewidth=0.2)
# plt.xticks(fontsize=8)
#
# for index, row in corr_df.iterrows():
#     y_offset = 0.01 if row['Correlation'] >= 0 else -0.01
#     plt.text(x=index, y=row['Correlation'] + y_offset, s=f"{row['Correlation']:.3f}",
#              ha='center', va='bottom' if row['Correlation'] >= 0 else 'top', fontsize=8)
#
# plt.legend([], [], frameon=False)
#
# plt.tight_layout()
# plt.show()
#
# plt.close()

# corr_df.to_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/F25/ECS170/ChessAI/ann_group/feature_corr2.csv',index=False)


