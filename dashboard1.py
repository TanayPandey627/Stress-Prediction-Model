import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

import joblib

base_models = joblib.load("base_models.pkl")
scaler = joblib.load("scaler.pkl")

# PAGE CONFIG
st.set_page_config(
    page_title="NeuroTrack",
    layout="wide",
    page_icon="🧠"
)

# 🎨 CALM UI THEME (SAGE + BLUE)
st.markdown("""
<style>

/* Background */
.main {
    background-color: #f5f7f6;
}

/* Text */
h1, h2, h3, p {
    color: ffffff;
}

/* Cards */
.metric-card {
    padding: 20px;
    border-radius: 16px;
    background-color: #e6f4ea;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    h1, h2, h3, p {
        color: #000000;
        }
}

.metric-card-blue {
    padding: 20px;
    border-radius: 16px;
    background-color: #e0f2fe;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    h1, h2, h3, p {
        color: #000000;
        }
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #60a5fa, #86efac);
    color: #1f2937;
    border-radius: 10px;
    height: 48px;
    font-size: 16px;
    border: none;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #ecfdf5;
    h1, h2, h3, p {
            color: #000000;
            }
}

</style>
""", unsafe_allow_html=True)


# HEADER
st.markdown("""
# 🧠 Interpretable Stacked Ensemble for Personlized Stress Prediction
### Personalized Stress Intelligence Platform
""")

st.caption("Calm UI • Explainable AI • Personalized Insights")


# SIDEBAR

st.sidebar.title("⚙️ Controls")

model_choice = st.sidebar.selectbox(
    "Meta Model",
    ["ElasticNet", "XGBoost", "CatBoost", "LightGBM"]
)

input_type = st.sidebar.radio(
    "Input Method",
    ["Manual Input (3 Days)", "Upload CSV"]
)


# MODELS

def get_base_models():
    return [
        SVR(kernel="linear"),
        Lasso(alpha=0.1),
        RandomForestRegressor(n_estimators=100)
    ]

def get_meta_model(name):
    
    if name == "ElasticNet":
        return joblib.load("elastic.pkl")

    elif name == "XGBoost":
        return joblib.load("xgb.pkl")

    elif name == "CatBoost":
        return joblib.load("cat.pkl")

    elif name == "LightGBM":
        return joblib.load("lgbm.pkl")

# INPUT SECTION

st.markdown("## 📥 Behavioral Input (3-Day Average)")

if input_type == "Manual Input (3 Days)":

    def input_3day(feature):
        c1, c2, c3 = st.columns(3)

        with c1:
            d1 = st.number_input(f"{feature} - Day 1 (min)", 0.0, key=feature+"1")
        with c2:
            d2 = st.number_input(f"{feature} - Day 2 (min)", 0.0, key=feature+"2")
        with c3:
            d3 = st.number_input(f"{feature} - Day 3 (min)", 0.0, key=feature+"3")

        return (d1 + d2 + d3) / 3
    
    screen = input_3day("Screen Time (minutes)")
    conv = input_3day("Conversation (minutes)")
    mobility = input_3day("Mobility (km)")
    dark = input_3day("Dark Time (hours)")
    stress_avg = input_3day("Recent Stress Level")

    input_data = pd.DataFrame([[
        screen,
        conv,
        mobility,
        dark,
        stress_avg
    ]], columns=[
        "screen_time_total",
        "average_conversation_duration",
        "total_distance_km",
        "avg_dark_time",
        "stress_3day_avg"
    ])

    # screen = input_3day("Screen Time")
    # conv = input_3day("Conversation")
    # mobility = input_3day("Mobility")
    # sleep = input_3day("Sleep")
    # dark = input_3day("Dark Time")

    # Baseline from 3-day data
    # baseline = np.mean([screen, conv, mobility, sleep, dark])

    # input_data = pd.DataFrame([[screen, conv, mobility, sleep, dark]])

else:
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        input_data = pd.read_csv(uploaded_file)

        # Ensure required columns exist
        required_cols = [
            "screen_time_total",
            "average_conversation_duration",
            "total_distance_km",
            "avg_dark_time",
            "stress_3day_avg"
        ]

        input_data = input_data[required_cols]

        # Use correct baseline
        stress_avg = input_data["stress_3day_avg"].iloc[0]

    else:
        input_data = None
        stress_avg = None


# PREDICTION

if st.button("🚀 Analyze Stress", use_container_width=True):

    if input_data is None:
        st.warning("Please provide input data")
    else:

        # LOAD META MODEL (based on user)
        meta_model = get_meta_model(model_choice)


        # TRANSFORM INPUT

        X_input = scaler.transform(input_data)

        # CREATE META FEATURES

        meta_input = np.zeros((input_data.shape[0], len(base_models)))

        for i, model in enumerate(base_models):
            meta_input[:, i] = model.predict(X_input)

        # FINAL PREDICTION

        pred = meta_model.predict(meta_input)[0]


        personalized = pred - stress_avg

        if personalized < -0.5:
            risk = "Low"
            color = "#22c55e"
        elif personalized < 0.5:
            risk = "Medium"
            color = "#eab308"
        else:
            risk = "High"
            color = "#ef4444"



        
        # OUTPUT
        
        st.markdown("## 📊 Insights")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(f"""
            <div class="metric-card">
            <h4>Predicted Stress</h4>
            <h2>{pred:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card-blue">
            <h4>Baseline (3-Day Avg)</h4>
            <h2>{stress_avg:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card">
            <h4>Risk Level</h4>
            <h2 style="color:{color}">{risk}</h2>
            </div>
            """, unsafe_allow_html=True)

        
        # VISUAL
        
        st.markdown("## 📈 Stress Comparison")

        chart = pd.DataFrame({
            "Type": ["Predicted", "Baseline"],
            "Value": [pred, stress_avg]
        })

        st.bar_chart(chart.set_index("Type"))

        
        # EXPLANATION
        
        st.markdown("## 🧠 AI Insights")

        try:
    # Using RandomForest from base models (stable for SHAP)
            rf_model = base_models[-1]

            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(X_input)

            feature_names = [
                "Screen Time",
                "Conversation",
                "Mobility",
                "Dark Time",
                "Recent Stress"
            ]

            shap_df = pd.DataFrame({
                "Feature": feature_names,
                "Impact": shap_values[0]
            })

            shap_df["abs"] = shap_df["Impact"].abs()
            shap_df = shap_df.sort_values("abs", ascending=True)

            # 📊 Plot
            fig, ax = plt.subplots()

            ax.barh(shap_df["Feature"], shap_df["Impact"])
            ax.set_title("Feature Contribution to Stress")
            ax.set_xlabel("Impact on Prediction")

            st.pyplot(fig)

            # 🧾 Text explanation
            st.markdown("### 🔍 Key Factors")

            top = shap_df.sort_values("abs", ascending=False).head(3)

            for _, row in top.iterrows():
                direction = "increased" if row["Impact"] > 0 else "reduced"
                st.write(f"- {row['Feature']} **{direction}** your stress")

        except Exception as e:
            st.warning("SHAP explanation could not be generated.")

        st.success("Analysis complete ✔")