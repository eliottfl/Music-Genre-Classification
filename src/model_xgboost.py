import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

df = pd.read_csv("data\\features_var.csv")

x = df.drop(columns=["genre", "track_id"])
y = df["genre"]

#label must be encoded for xgboost to map each class to the specific index in the future probability vector
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=0, stratify=y_encoded)

fig1, ax1 = plt.subplots(figsize=(10, 6))

def train_with_early_stopping():
    """Trains XGBoost model with an early stopping of 20 rounds and returns the model"""

    xgb_model = xgb.XGBClassifier(
        n_estimators=500, #500 can seem too high but early stopping will obviously stop before
        learning_rate=0.1, #slow down learning but it's more precise/robust
        max_depth=50,
        objective="multi:softmax", #multiples classes (>2), softmax give the winning genre
        num_class=len(label_encoder.classes_),
        early_stopping_rounds=20, #avoid overfitting after 20 iterations without mlogloss decreasing, we stop the training
        eval_metric="mlogloss", 
        random_state=0,
    )

    eval_set = [(x_train, y_train), (x_test, y_test)]
    
    #training
    xgb_model.fit(x_train, y_train, eval_set=eval_set)
    
    results = xgb_model.evals_result()
    
    #results["validation_0"]["mlogloss"] contains all the mlogloss values at each new tree
    # validation_0 = train, validation_1 = test
    epochs = len(results["validation_0"]["mlogloss"])
    
    ax1.set_title("Optimisation of XGBoost : Log Loss (Train vs Test)")
    ax1.plot(np.arange(epochs), results["validation_0"]["mlogloss"], label="Train Loss", color="grey", linestyle="--")
    ax1.plot(np.arange(epochs), results["validation_1"]["mlogloss"], label="Test Loss", color="b")
    ax1.axvline(x=xgb_model.best_iteration, color="r", linestyle="--", label=f"Best Iteration = {xgb_model.best_iteration}")
    ax1.set_xlabel("Number of trees")
    ax1.set_ylabel("Log Loss")
    ax1.legend()
    ax1.grid()
    plt.tight_layout()
    plt.savefig("figures\\xgboost_analysis.png")
    
    return xgb_model

xgb_model = train_with_early_stopping()

y_pred_encoded = xgb_model.predict(x_test)
y_test_decoded = label_encoder.inverse_transform(y_test)
y_pred_decoded = label_encoder.inverse_transform(y_pred_encoded)

#confusion matrix
fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(16, 7))
ax2.set_title("Confusion Matrix - XGBoost")
conf_matrix = confusion_matrix(y_test_decoded, y_pred_decoded)
sns.heatmap(conf_matrix, cmap="flare", annot=True, fmt="d", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=ax2)
ax2.set_xlabel("Prediction")
ax2.set_ylabel("Truth")

print(f"Accuracy : {accuracy_score(y_test_decoded, y_pred_decoded)}")

#feature importance
importances = xgb_model.feature_importances_
feature_names = x.columns
indices = np.argsort(importances)[-10:]

ax3.set_title("Top 10 most important features")
ax3.barh(np.arange(len(indices)), importances[indices])
ax3.set_yticks(np.arange(len(indices)))
ax3.set_yticklabels(feature_names[indices])
ax3.set_xlabel("Relative Importance")
ax3.set_axisbelow(True)
ax3.grid(axis='x')

plt.tight_layout()
plt.savefig("figures\\xgboost.png")