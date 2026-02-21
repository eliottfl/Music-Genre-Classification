import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data\\features_var.csv")

x = df.drop(columns=["genre", "track_id"])
y = df["genre"]

#80% of data for training, 20% for testing
#random_state fixed to ensure identical separation of data between training and testing for each execution, it ensures reproducible results
#stratify to ensure same proportion of genre is respected (for each genre, 80% is in train and 20% is in test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)

#normalization without taking information from test data
x_max = x_train.max()
x_min = x_train.min()
x_train = (x_train - x_min) / (x_max - x_min)
x_test = (x_test - x_min) / (x_max - x_min)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

def best_k_value(n):
    """Find the k with the best accuracy for kNN model and creates a figure to compare all the k tested"""
    accuracies = [0 for _ in range(n)]

    for k in range(1,n+1):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)

        y_pred = knn.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies[k-1] = accuracy

    best_acc = max(accuracies)
    best_k = accuracies.index(best_acc) + 1

    ax1.set_title("Accuracy depending of k for kNN model")
    ax1_x = [i for i in range(1, n+1)]
    ax1.plot(ax1_x, accuracies)
    ax1.set_xlabel("k")
    ax1.set_ylabel("Accuracy")
    ax1.axvline(x=best_k, color="r", linestyle="--", label=f"Best k = {best_k}")
    ax1.axhline(y=best_acc, color="g", linestyle="--", label=f"Best accuracy = {np.round(best_acc, 4)}")
    ax1.grid()
    ax1.legend()
    return (best_k)

k = best_k_value(100)
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

#confusion matrix
ax2.set_title("Confusion matrix - kNN")
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, cmap="flare", annot=True, fmt="d", xticklabels=knn.classes_, yticklabels=knn.classes_, ax=ax2)
ax2.set_xlabel("Prediction")
ax2.set_ylabel("Truth")
plt.tight_layout()
plt.savefig("figures\\knn.png")