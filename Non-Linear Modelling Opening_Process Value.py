# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import curve_fit

st.set_page_config(layout="wide")
st.title("Non-linear Modeling: Valve Opening vs Process Value")

# Upload data
uploaded_file = st.file_uploader("Upload CSV with 'Opening' and 'Process' columns", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Raw Data", data.head())

    if 'Opening' in data.columns and 'Process' in data.columns:
        x = data['Opening'].values.reshape(-1, 1)
        y = data['Process'].values

        st.subheader("Model Comparison and Curve Fitting")

        models = {}

        # Linear Model
        lin_model = LinearRegression().fit(x, y)
        lin_pred = lin_model.predict(x)
        models["Linear"] = {
            "pred": lin_pred,
            "r2": r2_score(y, lin_pred),
            "eq": f"y = {lin_model.coef_[0]:.4f}*x + {lin_model.intercept_:.4f}",
            "gain": lin_model.coef_[0]
        }

        # Polynomial (degree 2 and 3)
        for d in [2, 3]:
            poly = PolynomialFeatures(degree=d)
            x_poly = poly.fit_transform(x)
            model = LinearRegression().fit(x_poly, y)
            pred = model.predict(x_poly)
            coeffs = model.coef_
            eq = "y = " + " + ".join([f"{coeffs[i]:.4f}*x^{i}" for i in range(len(coeffs))])
            models[f"Poly (deg {d})"] = {
                "pred": pred,
                "r2": r2_score(y, pred),
                "eq": eq,
                "gain": coeffs[1]  # First derivative approx at origin
            }

        # Exponential Model
        try:
            def exp_func(x, a, b, c): return a * np.exp(b * x) + c
            popt, _ = curve_fit(exp_func, x.flatten(), y, maxfev=10000)
            exp_pred = exp_func(x.flatten(), *popt)
            models["Exponential"] = {
                "pred": exp_pred,
                "r2": r2_score(y, exp_pred),
                "eq": f"y = {popt[0]:.4f} * e^({popt[1]:.4f}*x) + {popt[2]:.4f}",
                "gain": popt[0] * popt[1]
            }
        except:
            st.warning("Exponential model failed to converge.")

        # Logarithmic Model
        try:
            def log_func(x, a, b): return a * np.log(x) + b
            x_log = x[x > 0]
            y_log = y[:len(x_log)]
            popt, _ = curve_fit(log_func, x_log.flatten(), y_log)
            log_pred = log_func(x.flatten(), *popt)
            models["Logarithmic"] = {
                "pred": log_pred,
                "r2": r2_score(y, log_pred),
                "eq": f"y = {popt[0]:.4f} * ln(x) + {popt[1]:.4f}",
                "gain": popt[0]
            }
        except:
            st.warning("Logarithmic model failed (x must be > 0).")

        # Trigonometric Model (Sinusoidal Fit)
        try:
            def trig_func(x, a, b, c): return a * np.sin(b * x) + c
            popt, _ = curve_fit(trig_func, x.flatten(), y)
            trig_pred = trig_func(x.flatten(), *popt)
            models["Trigonometric"] = {
                "pred": trig_pred,
                "r2": r2_score(y, trig_pred),
                "eq": f"y = {popt[0]:.4f} * sin({popt[1]:.4f}*x) + {popt[2]:.4f}",
                "gain": popt[0] * popt[1]  # Approx slope
            }
        except:
            st.warning("Trigonometric model fitting failed.")

        # Results Table
        st.subheader("📊 Model Summary")
        result_df = pd.DataFrame([
            {
                "Model": name,
                "R² Score": info["r2"],
                "Gain": round(info["gain"], 4),
                "Equation": info["eq"]
            }
            for name, info in models.items()
        ]).sort_values("R² Score", ascending=False)

        st.dataframe(result_df)

        # Plot best model
        best_model = result_df.iloc[0]["Model"]
        st.success(f"✅ Best Model: **{best_model}**")

        fig, ax = plt.subplots()
        ax.scatter(x, y, label='Data', color='black')
        for name, info in models.items():
            ax.plot(x.flatten(), info["pred"], label=f"{name} (R²={info['r2']:.3f})")
        ax.set_xlabel("Valve Opening (%)")
        ax.set_ylabel("Process Value")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.error("CSV must have 'Opening' and 'Process' columns.")
