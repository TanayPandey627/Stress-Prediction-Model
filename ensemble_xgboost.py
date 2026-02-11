import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

# === Load and preprocess ===
df = pd.read_csv("Final.csv", parse_dates=["date"])
df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
df = df.sort_values(["user_id", "date"])

df["user_mean_stress"] = df.groupby("user_id")["stress_level"].apply(lambda x: x.shift().expanding().mean()).reset_index(level=0, drop=True)
df["prev_3day_avg_stress"] = df.groupby("user_id")["stress_level"].apply(lambda x: x.shift().rolling(window=3, min_periods=1).mean()).reset_index(level=0, drop=True)

X = df.drop(columns=["user_id", "date", "stress_level", "stress_3day_avg"], errors="ignore")
y = df["stress_level"]
groups = df["user_id"]

X = X.fillna(0)
y = y.fillna(y.mean())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Models ===
base_models = {
    "svr": SVR(kernel='linear', C=1.0),
    "lasso": Lasso(alpha=0.1),
    "rf": RandomForestRegressor(n_estimators=100, random_state=42)
}
meta_model = XGBRegressor(n_estimators=100, random_state=42)

# === Set up GroupKFold ===
gkf = GroupKFold(n_splits=5)

# === Prepare meta features ===
meta_X = np.zeros((X.shape[0], len(base_models)))
meta_y = y.values

for fold, (train_idx, test_idx) in enumerate(gkf.split(X_scaled, y, groups)):
    print(f"Fold {fold + 1}")
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train = y.iloc[train_idx]

    for i, (name, model) in enumerate(base_models.items()):
        model.fit(X_train, y_train)
        meta_X[test_idx, i] = model.predict(X_test)

# === Train meta model ===
meta_model.fit(meta_X, meta_y)
final_pred = meta_model.predict(meta_X)

# === Evaluate
mae = mean_absolute_error(meta_y, final_pred)
print(f"\n Stacked Model MAE: {round(mae, 4)}")

# === Optional plot
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.plot(meta_y, label="Actual", marker='o')
plt.plot(final_pred, label="Predicted", linestyle='--', marker='x')
plt.xlabel("Sample Index")
plt.ylabel("Stress Level")
plt.title("📈 Actual vs Predicted (Stacked Model)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
