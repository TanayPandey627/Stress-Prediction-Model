import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("Final.csv")

# print(df.columns.tolist())

selected_features = [
    "screen_time_total",
    "average_conversation_duration",
    "total_distance_km",
    "avg_dark_time",
    "stress_3day_avg"
]

X = df[selected_features]
y = df["stress_level"]

X.fillna(0, inplace=True)

# ===============================
# SCALER
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# BASE MODELS
# ===============================
base_models = [
    SVR(kernel="linear"),
    Lasso(alpha=0.1),
    RandomForestRegressor(n_estimators=200, random_state=42)
]

meta_features = np.zeros((X_scaled.shape[0], len(base_models)))

for i, model in enumerate(base_models):
    model.fit(X_scaled, y)
    meta_features[:, i] = model.predict(X_scaled)

# ===============================
# META MODELS
# ===============================
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
xgb = XGBRegressor(n_estimators=200, random_state=42)
cat = CatBoostRegressor(verbose=0, random_state=42)
lgbm = LGBMRegressor(n_estimators=200, random_state=42)

# Train all
elastic.fit(meta_features, y)
xgb.fit(meta_features, y)
cat.fit(meta_features, y)
lgbm.fit(meta_features, y)

# ===============================
# SAVE EVERYTHING
# ===============================
joblib.dump(base_models, "base_models.pkl")
joblib.dump(scaler, "scaler.pkl")

joblib.dump(elastic, "elastic.pkl")
joblib.dump(xgb, "xgb.pkl")
joblib.dump(cat, "cat.pkl")
joblib.dump(lgbm, "lgbm.pkl")

print("✅ All models saved successfully!")