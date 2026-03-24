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

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="NeuroTrack",
    layout="wide",
    page_icon="🧠"
)

# ===============================
# 🎨 CALM UI THEME (SAGE + BLUE)
# ===============================
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


# ===============================
# HEADER
# ===============================
st.markdown("# 🧠 NeuroTrack")
st.markdown("### Personalized Stress Intelligence Platform")
st.caption("Calm UI • Explainable AI • Personalized Insights")

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("⚙️ Controls")

model_choice = st.sidebar.selectbox(
    "Meta Model",
    ["ElasticNet", "XGBoost", "CatBoost", "LightGBM"]
)

input_type = st.sidebar.radio(
    "Input Method",
    ["Manual Input (3 Days)", "Upload CSV"]
)

# ===============================
# MODELS
# ===============================
def get_base_models():
    return [
        SVR(kernel="linear"),
        Lasso(alpha=0.1),
        RandomForestRegressor(n_estimators=100)
    ]

def get_meta_model(name):
    if name == "ElasticNet":
        return ElasticNet(alpha=0.1, l1_ratio=0.5)
    elif name == "XGBoost":
        return XGBRegressor(n_estimators=100)
    elif name == "CatBoost":
        return CatBoostRegressor(verbose=0)
    elif name == "LightGBM":
        return LGBMRegressor()

# ===============================
# INPUT SECTION
# ===============================
st.markdown("## 📥 Behavioral Input (3-Day Average)")

if input_type == "Manual Input (3 Days)":

    def input_3day(feature):
        c1, c2, c3 = st.columns(3)
        with c1:
            d1 = st.number_input(f"{feature} Day 1", 0.0, key=feature+"1")
        with c2:
            d2 = st.number_input(f"{feature} Day 2", 0.0, key=feature+"2")
        with c3:
            d3 = st.number_input(f"{feature} Day 3", 0.0, key=feature+"3")
        return (d1 + d2 + d3) / 3

    screen = input_3day("Screen Time (minutes)")
    conv = input_3day("Conversation (minutes)")
    mobility = input_3day("Mobility")
    sleep = input_3day("Sleep (hours)")
    dark = input_3day("Dark Time (hours)")

    baseline = np.mean([screen, conv, mobility, sleep, dark])

    input_data = pd.DataFrame([[screen, conv, mobility, sleep, dark]])

else:
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        input_data = pd.read_csv(uploaded_file)
        baseline = input_data.mean().mean()
    else:
        input_data = None
        baseline = None

# ===============================
# PREDICTION
# ===============================
if st.button("🚀 Analyze Stress", use_container_width=True):

    if input_data is None:
        st.warning("Please provide input data")
    else:

        # Dummy training
        X_train = np.random.rand(100, input_data.shape[1])
        y_train = np.random.rand(100)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_input = scaler.transform(input_data)

        base_models = get_base_models()

        meta_train = np.zeros((100, len(base_models)))
        meta_input = np.zeros((input_data.shape[0], len(base_models)))

        for i, model in enumerate(base_models):
            model.fit(X_train, y_train)
            meta_train[:, i] = model.predict(X_train)
            meta_input[:, i] = model.predict(X_input)

        meta_model = get_meta_model(model_choice)
        meta_model.fit(meta_train, y_train)

        pred = meta_model.predict(meta_input)[0]
        personalized = pred - baseline

        if personalized < -0.5:
            risk = "Low"
            color = "#22c55e"
        elif personalized < 0.5:
            risk = "Medium"
            color = "#eab308"
        else:
            risk = "High"
            color = "#ef4444"

        # ===============================
        # OUTPUT
        # ===============================
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
            <h2>{baseline:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card">
            <h4>Risk Level</h4>
            <h2 style="color:{color}">{risk}</h2>
            </div>
            """, unsafe_allow_html=True)

        # ===============================
        # VISUAL
        # ===============================
        st.markdown("## 📈 Stress Comparison")

        chart = pd.DataFrame({
            "Type": ["Predicted", "Baseline"],
            "Value": [pred, baseline]
        })

        st.bar_chart(chart.set_index("Type"))

        # ===============================
        # SHAP EXPLAINABILITY
        # ===============================
        st.markdown("## 🧠 Explainable AI (SHAP)")

        try:
            explainer_model = RandomForestRegressor(n_estimators=100)
            explainer_model.fit(X_train, y_train)

            explainer = shap.TreeExplainer(explainer_model)
            shap_values = explainer.shap_values(X_input)

            feature_names = ["screen", "conversation", "mobility", "sleep", "dark"]

            shap_df = pd.DataFrame({
                "Feature": feature_names,
                "Impact": shap_values[0]
            })

            shap_df["abs"] = shap_df["Impact"].abs()
            shap_df = shap_df.sort_values("abs", ascending=True)

            fig, ax = plt.subplots()
            ax.barh(shap_df["Feature"], shap_df["Impact"])
            ax.set_title("Feature Contribution")
            ax.set_xlabel("Impact")

            st.pyplot(fig)

            # TEXT EXPLANATION
            st.markdown("### 🔍 Key Drivers")

            top = shap_df.sort_values("abs", ascending=False).head(3)

            for _, row in top.iterrows():
                direction = "increased" if row["Impact"] > 0 else "reduced"
                st.write(f"- {row['Feature']} {direction} the stress level")

        except:
            st.warning("SHAP explanation could not be generated.")

        st.success("Analysis complete ✔")