
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.svm import SVR
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

# PLOT STYLE SETTINGS

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("talk")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["lines.linewidth"] = 2

# LOAD & PREPROCESS

df = pd.read_csv("Final.csv", parse_dates=["date"])
df = df.sort_values(["user_id", "date"])

df["stress_level"] = pd.to_numeric(df["stress_level"], errors="coerce")
df["stress_level"] = df["stress_level"].fillna(df["stress_level"].mean())

df["user_mean_stress"] = (
    df.groupby("user_id")["stress_level"]
    .apply(lambda x: x.shift().expanding().mean())
    .reset_index(level=0, drop=True)
)

all_features = df.drop(
    columns=["user_id", "date", "stress_level"],
    errors="ignore"
)

behavioral_cols = [
    col for col in all_features.columns
    if col not in [
        "user_mean_stress",
        "prev_3day_avg_stress",
        "stress_3day_avg"
    ]
]

X = all_features.fillna(0)
X_behavioral = all_features[behavioral_cols].fillna(0)

y = df["stress_level"]
groups = df["user_id"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# BASE MODELS

base_models = [
    SVR(kernel="linear", C=1.0),
    Lasso(alpha=0.1),
    RandomForestRegressor(n_estimators=200, random_state=42)
]

# META MODELS

meta_models = {
    "XGBoost": XGBRegressor(n_estimators=200, random_state=42),
    "CatBoost": CatBoostRegressor(iterations=200, random_state=42, verbose=False),
    "LightGBM": LGBMRegressor(n_estimators=200, random_state=42),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5)
}

# NESTED GROUPKFOLD EVALUATION

outer_gkf = GroupKFold(n_splits=5)
results = {}
cv_predictions = np.zeros(len(y))

for meta_name, meta_model in meta_models.items():

    fold_mae = []
    fold_rmse = []

    for train_idx, test_idx in outer_gkf.split(X_scaled, y, groups):

        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]

        inner_gkf = GroupKFold(n_splits=4)

        meta_train = np.zeros((len(train_idx), len(base_models)))
        meta_test = np.zeros((len(test_idx), len(base_models)))

        for i, model in enumerate(base_models):

            oof_preds = np.zeros(len(train_idx))

            for inner_train, inner_val in inner_gkf.split(
                X_train, y_train, groups_train
            ):
                model.fit(X_train[inner_train], y_train.iloc[inner_train])
                oof_preds[inner_val] = model.predict(X_train[inner_val])

            meta_train[:, i] = oof_preds

            model.fit(X_train, y_train)
            meta_test[:, i] = model.predict(X_test)

        meta_model.fit(meta_train, y_train)
        preds = meta_model.predict(meta_test)

        fold_mae.append(mean_absolute_error(y_test, preds))
        fold_rmse.append(np.sqrt(mean_squared_error(y_test, preds)))

        if meta_name == "ElasticNet":
            cv_predictions[test_idx] = preds

    results[meta_name] = {
        "MAE": np.mean(fold_mae),
        "RMSE": np.mean(fold_rmse),
        "STD": np.std(fold_mae)
    }

comparison_df = pd.DataFrame(results).T.round(4)
print("\n📊 FINAL META-MODEL COMPARISON")
print(comparison_df)

# HIGH-QUALITY META MODEL COMPARISON PLOT

comparison_df.plot(kind="bar", colormap="viridis")
plt.title("Meta-Model Performance Comparison")
plt.ylabel("Error")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# CROSS-VALIDATED ACTUAL vs PREDICTED

plt.figure()
plt.plot(y.values[:100], label="Actual")
plt.plot(cv_predictions[:100], linestyle="--", label="Predicted")
plt.title("Cross-Validated Actual vs Predicted Stress")
plt.xlabel("Sample Index")
plt.ylabel("Stress Level")
plt.legend()
plt.tight_layout()
plt.show()

# Scatter plot
plt.figure()
plt.scatter(y, cv_predictions, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.xlabel("Actual Stress")
plt.ylabel("Predicted Stress")
plt.title("Actual vs Predicted (Scatter)")
plt.tight_layout()
plt.show()

# FINAL MODEL FOR DEPLOYMENT

final_meta_features = np.zeros((X_scaled.shape[0], len(base_models)))

for i, model in enumerate(base_models):
    model.fit(X_scaled, y)
    final_meta_features[:, i] = model.predict(X_scaled)

final_meta_model = ElasticNet(alpha=0.1, l1_ratio=0.5)
final_meta_model.fit(final_meta_features, y)

df["predicted_stress"] = final_meta_model.predict(final_meta_features)

# PERSONALIZATION & CLASSIFICATION

df["personalized_stress"] = (
    df["predicted_stress"] - df["user_mean_stress"]
)

def stress_label(x):
    if x < -0.5:
        return "Low"
    elif x < 0.5:
        return "Medium"
    return "High"

df["stress_category"] = df["personalized_stress"].apply(stress_label)

# SHAP (BEHAVIORAL FEATURES ONLY)

scaler_beh = StandardScaler()
X_beh_scaled = scaler_beh.fit_transform(X_behavioral)

rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_beh_scaled, y)

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_beh_scaled)

# Global importance
shap.summary_plot(
    shap_values,
    X_behavioral,
    plot_type="bar",
    show=True
)

# Detailed beeswarm
shap.summary_plot(
    shap_values,
    X_behavioral,
    show=True
)


#  HUMAN-READABLE EXPLANATION


sample_idx = 0
shap_vals_user = shap_values[sample_idx]

feature_contrib = pd.DataFrame({
    "feature": X_behavioral.columns,
    "shap_value": shap_vals_user
})

feature_contrib["abs_val"] = feature_contrib["shap_value"].abs()
top_features = feature_contrib.sort_values(
    "abs_val", ascending=False
).head(3)

print("\n Text Explanation (Behavioral Causes):")
for _, row in top_features.iterrows():
    direction = "increased" if row["shap_value"] > 0 else "reduced"
    print(f"- {row['feature']} {direction} the predicted stress level")

