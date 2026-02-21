import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data\\features_corr.csv")

threshold = 0.005

genre_col = df["genre"]
track_id_col = df["track_id"]

df_features = df.drop(columns=["genre", "track_id"])

#normalization
df_min = df_features.min()
df_max = df_features.max()
df_norm = (df_features - df_min) / (df_max - df_min)

variances = df_norm.var()

to_drop=[]
for col in variances.index:
    if variances[col] < threshold:
        to_drop.append(col)

plt.figure(figsize=(14, 12))
plt.title("Variances of features")
plt.bar(variances.index, variances.values)
plt.xticks(rotation=45, ha="right")
plt.axhline(y=threshold, color="red", label=f"Threshold ({threshold})")
plt.ylabel("Variance")
plt.xlabel("Features")
plt.savefig("figures\\variance.png")

df_filtered = df.drop(columns=to_drop)
print(f"Columns to drop : {to_drop}")
df_filtered.to_csv("data\\features_var.csv", index=False)