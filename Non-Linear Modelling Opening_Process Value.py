import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from fpdf import FPDF
import io
from datetime import datetime

st.set_page_config(layout="wide")
st.title("🧠 Non-linear Modeling Tool for Valve Opening vs Process Value")

# === Upload Section ===
uploaded_file = st.file_uploader("📁 Upload CSV with 'Opening' and 'Process' columns", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### 🔍 Uploaded Data Preview", data.head())

    if 'Opening' in data.columns and 'Process' in data.columns:
        x = data['Opening'].values.reshape(-1, 1)
        y = data['Process'].values

        # === Model Fitting ===
        models = {}
        lin_model = LinearRegression().fit(x, y)
        lin_pred = lin_model.predict(x)
        models["Linear"] = {
            "pred": lin_pred,
            "r2": r2_score(y, lin_pred),
            "eq": f"y = {lin_model.coef_[0]:.4f}*x + {lin_model.intercept_:.4f}",
            "gain": lin_model.coef_[0]
        }

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
                "gain": coeffs[1]
            }

        try:
            def exp_func(x, a, b, c): return a * np.exp(b * x) + c
            popt, _ = curve_fit(exp_func, x.flatten(), y, maxfev=10000)
            exp_pred = exp_func(x.flatten(), *popt)
            models["Exponential"] = {
                "pred": exp_pred,
                "r2": r2_score(y, exp_pred),
                "eq": f"y = {popt[0]:.4f}*e^({popt[1]:.4f}*x)+{popt[2]:.4f}",
                "gain": popt[0] * popt[1]
            }
        except:
            pass

        try:
            def log_func(x, a, b): return a * np.log(x) + b
            valid_x = x[x > 0]
            valid_y = y[:len(valid_x)]
            popt, _ = curve_fit(log_func, valid_x.flatten(), valid_y)
            log_pred = log_func(x.flatten(), *popt)
            models["Logarithmic"] = {
                "pred": log_pred,
                "r2": r2_score(y, log_pred),
                "eq": f"y = {popt[0]:.4f}*ln(x) + {popt[1]:.4f}",
                "gain": popt[0]
            }
        except:
            pass

        # === Gain Scheduling ===
        st.subheader("📈 Gain Scheduling Visualization")
        local_gain = np.gradient(y, x.flatten())
        fig1, ax1 = plt.subplots()
        ax1.plot(x, local_gain, label='Estimated Gain')
        ax1.set_xlabel("Opening (%)")
        ax1.set_ylabel("Gain (dy/dx)")
        ax1.set_title("Gain vs Valve Opening")
        ax1.grid(True)
        st.pyplot(fig1)

        # === Best Model Selection ===
        result_df = pd.DataFrame([
            {"Model": name, "R2": info['r2'], "Gain": round(info['gain'], 4), "Equation": info['eq']}
            for name, info in models.items()
        ]).sort_values("R2", ascending=False)

        st.subheader("🏁 Model Summary")
        st.dataframe(result_df)
        best_model = result_df.iloc[0]['Model']
        st.success(f"Best model: {best_model}")

        # === Step Response Simulation ===
        st.subheader("🚀 Simulated Step Response")
        col1, col2 = st.columns(2)
        with col1:
            system_type = st.selectbox("System Type", ["First Order", "Second Order"])
            K = st.number_input("Gain (K)", 1.0)
        with col2:
            tau = st.number_input("Time Constant (τ) [First Order]", 5.0)
            wn = st.number_input("ωn (Second Order Natural Freq)", 1.0)
            zeta = st.slider("ζ (Damping Ratio)", 0.0, 2.0, 0.7, 0.01)

        t = np.linspace(0, 20, 500)
        if system_type == "First Order":
            y_step = K * (1 - np.exp(-t / tau))
        else:
            wd = wn * np.sqrt(1 - zeta**2) if zeta < 1 else 0
            y_step = 1 - np.exp(-zeta * wn * t) * (
                np.cos(wd * t) + (zeta / np.sqrt(1 - zeta**2)) * np.sin(wd * t)) if zeta < 1 else 1 - np.exp(-wn * t)
            y_step *= K

        fig2, ax2 = plt.subplots()
        ax2.plot(t, y_step, label=system_type)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Process Output")
        ax2.grid(True)
        ax2.set_title(f"Simulated Step Response: {system_type}")
        st.pyplot(fig2)

        # === Export PDF ===
        def create_pdf():
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "Modeling Report", ln=1, align='C')
            pdf.set_font("Arial", '', 12)
            pdf.cell(0, 10, f"Best Model: {best_model}", ln=1)
            pdf.multi_cell(0, 10, f"Equation: {models[best_model]['eq']}")
            pdf.cell(0, 10, f"R²: {models[best_model]['r2']:.4f}", ln=1)
            pdf.cell(0, 10, f"Gain: {models[best_model]['gain']:.4f}", ln=1)
            pdf_output = io.BytesIO()
            pdf.output(pdf_output)
            pdf_output.seek(0)
            return pdf_output

        st.download_button("📄 Download PDF Report", create_pdf(), file_name="model_report.pdf")

        # === Export Excel ===
        def to_excel():
            output = io.BytesIO()
            writer = pd.ExcelWriter(output, engine='openpyxl')
            data.to_excel(writer, index=False, sheet_name='Raw Data')
            result_df.to_excel(writer, index=False, sheet_name='Model Summary')
            for name, info in models.items():
                df = pd.DataFrame({"Opening": x.flatten(), "Prediction": info['pred']})
                df.to_excel(writer, index=False, sheet_name=name[:31])
            writer.close()
            output.seek(0)
            return output

        st.download_button("📊 Download Excel Report", to_excel(), file_name="model_results.xlsx")

    else:
        st.error("Uploaded file must contain 'Opening' and 'Process' columns.")
else:
    st.info("Awaiting CSV upload...")
