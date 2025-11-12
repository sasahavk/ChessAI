import pandas as pd
import matplotlib.pyplot as plt

# Load your results
df = pd.read_csv("results_log.csv")

# Count results
counts = df["winner"].value_counts().reindex(["white", "black", "draw"], fill_value=0)

# Plot a bar chart
plt.figure(figsize=(6, 4))
bars = plt.bar(counts.index, counts.values, color=["#66bb6a", "#ef5350", "#ffee58"])

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.1, int(height), ha='center', va='bottom')

plt.title("Game Results Summary")
plt.xlabel("Result")
plt.ylabel("Number of Games")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
