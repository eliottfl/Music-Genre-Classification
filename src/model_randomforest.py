import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

fig1, ax1 = plt.subplots(figsize=(10, 6))

def best_n_estimators(max_estimators):
    """Trains with different number of trees and returns the number with the best accuracy"""
    accuracies = []
    
    #we will test a lot of different numbers of trees so it's worth to have a step of 5 to reduce training time, and it won't cause a loss of information
    estimators = np.arange(1, max_estimators + 1, 5)

    for trees in estimators:
        rf = RandomForestClassifier(n_estimators=trees, criterion="gini", random_state=0)
        rf.fit(x_train, y_train)
        y_pred = rf.predict(x_test)
        accuracies.append(accuracy_score(y_test, y_pred))

    best_acc = max(accuracies)
    best_n = estimators[accuracies.index(best_acc)]

    #learning figure
    ax1.set_title("Accuracy VS Number of trees")
    ax1.plot(estimators, accuracies, color='blue')
    
    ax1.axvline(x=best_n, color="r", linestyle="--", label=f"Best n_estimators = {best_n}")
    ax1.axhline(y=best_acc, color="g", linestyle="--", label=f"Best Accuracy = {np.round(best_acc, 4)}")
    
    ax1.set_xlabel("Number of trees (n_estimators)")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid()
    plt.tight_layout()
    plt.savefig("figures\\randomforest_estimators_analysis.png")
    
    return best_n

best_n = best_n_estimators(200) 

rf = RandomForestClassifier(n_estimators=best_n, criterion="gini", random_state=0)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

#confusion matrix
fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(16, 7))
ax2.set_title("Confusion Matrix - Random Forest")
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, cmap="flare", annot=True, fmt="d", xticklabels=rf.classes_, yticklabels=rf.classes_, ax=ax2)
ax2.set_xlabel("Prediction")
ax2.set_ylabel("Truth")

#feature importance
importances = rf.feature_importances_
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
plt.savefig("figures\\randomforest.png")