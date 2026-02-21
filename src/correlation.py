import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data\\features.csv")

threshold = 0.95

genre = df["genre"]
track_id = df["track_id"]

#correlation is only tested on features
df_features = df.drop(columns=["genre", "track_id"])

#.abs() to detect inverse correlations
correlation = df_features.corr().abs()

#display
plt.figure(figsize=(16, 13))
sns.heatmap(correlation, cmap="flare")
plt.title("Correlation Matrix Heatmap")
plt.tight_layout()
plt.savefig("figures\\correlation_heatmap.png")

upper_matrix_true = np.triu(np.ones(correlation.shape), k=1).astype(bool)
correlation_upper = correlation.where(upper_matrix_true) #apply the filter matrix 

to_drop = []
for column in correlation_upper:
    for values in correlation_upper[column]:
        if values > threshold:
            to_drop.append(column)
            break

df_filtered = df.drop(columns=to_drop)
print(f"Columns to drop : {to_drop}")
df_filtered.to_csv("data\\features_corr.csv", index=False)