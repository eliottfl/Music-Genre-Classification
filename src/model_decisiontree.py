import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data\\features_var.csv")

x = df.drop(columns=["genre", "track_id"])
y = df["genre"]

#80% of data for training, 20% for testing
#random_state fixed to ensure identical separation of data between training and testing for each execution, it allows reproducilbe results
#stratify to ensure same proportion of genre is respected (for each genre, 80% is in train and 20% is in test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)

#no normalization needed contrary to knn

fig1, ax1 = plt.subplots(figsize=(10, 6))

def best_max_depth(max_depth):
    """Trains trees with differents depths to detect overfitting and returns depth with the best accuracy"""
    train_accuracies = []
    test_accuracies = []
    depths = np.arange(1, max_depth + 1)

    for depth in depths:
        
        #I learned ID3 algorithm in my lectures but I wanted to learn by myself and try/compare the Gini criterion because it's the most used in industry as I searched on internet
        dt = DecisionTreeClassifier(max_depth=depth, criterion="gini", random_state=0)
        dt.fit(x_train, y_train)

        train_accuracies.append(accuracy_score(y_train, dt.predict(x_train)))
        test_accuracies.append(accuracy_score(y_test, dt.predict(x_test)))

    best_acc = max(test_accuracies)
    best_depth = depths[test_accuracies.index(best_acc)]

    #learning figure
    ax1.set_title("Accuracy VS Depth | Overfitting detection")
    ax1.plot(depths, train_accuracies, label="Train Accuracy", linestyle="--", color="grey")
    ax1.plot(depths, test_accuracies, label="Test Accuracy", color="b")
    
    ax1.axvline(x=best_depth, color="r", linestyle="--", label=f"Best Depth = {best_depth}")
    ax1.axhline(y=best_acc, color="g", linestyle="--", label=f"Best Test Accuracy = {np.round(best_acc, 4)}")
    
    ax1.set_xlabel("Depth")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid()
    plt.tight_layout()
    plt.savefig("figures\\decisiontree_depth_analysis.png")

    return best_depth

best_depth = best_max_depth(20)

dt = DecisionTreeClassifier(max_depth=best_depth, criterion="gini", random_state=0)
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)

#confusion matrix
fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(16, 7))
ax2.set_title("Confusion matrix - Decision tree")
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, cmap="flare", annot=True, fmt="d", xticklabels=dt.classes_, yticklabels=dt.classes_, ax=ax2)
ax2.set_xlabel("Prediction")
ax2.set_ylabel("Truth")

#feature importance
importances = dt.feature_importances_
feature_names = x.columns
indices = np.argsort(importances)[-10:] #top 10 most important

ax3.set_title("Top 10 most important features")
ax3.barh(np.arange(len(indices)), importances[indices])
ax3.set_yticks(np.arange(len(indices)))
ax3.set_yticklabels(feature_names[indices])
ax3.set_xlabel("Relative Importance")
ax3.set_axisbelow(True)
ax3.grid(axis='x')

plt.tight_layout()
plt.savefig("figures\\decisiontree.png")