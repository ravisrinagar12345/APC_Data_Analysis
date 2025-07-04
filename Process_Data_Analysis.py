import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
import tempfile
import time
import io
from fpdf import FPDF
import xlsxwriter

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Advanced Process System Identification and Analysis")

# === Transformations ===
def generate_transforms(X_df):
    df_transformed = pd.DataFrame(index=X_df.index)
    for col in X_df.columns:
        x = X_df[col]
        df_transformed[f'{col}_orig'] = x
        df_transformed[f'{col}_abs'] = x.abs()
        df_transformed[f'{col}_exp'] = np.exp(np.clip(x, -50, 50))
        df_transformed[f'{col}_ln'] = np.where(x > 0, np.log(x), 0)
        df_transformed[f'{col}_log10'] = np.where(x > 0, np.log10(x), 0)
        df_transformed[f'{col}_sqrt'] = np.where(x >= 0, np.sqrt(x), 0)
        df_transformed[f'{col}_square'] = np.power(x, 2)
        df_transformed[f'{col}_cos'] = np.cos(x)
        df_transformed[f'{col}_sin'] = np.sin(x)
        df_transformed[f'{col}_tan'] = np.tan(x)
    df_transformed = df_transformed.replace([np.inf, -np.inf], 0).fillna(0)
    return df_transformed

# === Estimate order, delay, process time ===
def estimate_order_and_times(u, y, time_vector, y_pred=None):
    if y_pred is None:
        y_pred = y
    if np.ptp(u) != 0:
        u_norm = (u - np.min(u)) / np.ptp(u)
    else:
        u_norm = u
    if np.ptp(y_pred) != 0:
        y_norm = (y_pred - np.min(y_pred)) / np.ptp(y_pred)
    else:
        y_norm = y_pred
    delta_u = np.diff(u_norm)
    step_index = np.argmax(np.abs(delta_u)) if len(delta_u) > 0 else 0
    t = np.arange(len(u)) if time_vector is None else np.array((time_vector - time_vector[0]).total_seconds())
    delay_index = step_index
    for i in range(step_index, len(y_norm)):
        if abs(y_norm[i] - y_norm[step_index]) > 0.02:
            delay_index = i
            break
    delay_time = t[delay_index] - t[step_index] if len(t) > delay_index and len(t) > step_index else np.nan
    final_value = y_norm[-1]
    target_63 = y_norm[step_index] + 0.632 * (final_value - y_norm[step_index])
    process_time = np.nan
    for i in range(step_index, len(y_norm)):
        if y_norm[i] >= target_63:
            process_time = t[i] - t[step_index]
            break
    peaks, _ = find_peaks(np.gradient(np.gradient(y_norm)))
    order = 'First Order' if len(peaks) < 1 else 'Second Order'
    return {'order': order, 'delay_time': delay_time, 'process_time': process_time}

# === FFT frequency analysis ===
def analyze_fft(signal, sampling_rate, n_peaks=3):
    n = len(signal)
    if n == 0:
        return []
    freqs = fftfreq(n, d=1 / sampling_rate)
    fft_vals = fft(signal)
    mag = np.abs(fft_vals)
    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]
    mag = mag[pos_mask]
    indices = np.argsort(mag)[-n_peaks:]
    dominant_freqs = freqs[indices]
    return np.sort(dominant_freqs)

# === Core modeling and analysis ===
def run_analysis_with_relations(df, io_relations):
    df = df.dropna()
    results = []

    time_vector = df.index if df.index.dtype == 'datetime64[ns]' else None
    if time_vector is None or len(time_vector) < 2:
        sampling_rate = 1.0
    else:
        sampling_rate = 1 / ((time_vector[1] - time_vector[0]).total_seconds())

    all_inputs = set(i for inputs in io_relations.values() for i in inputs)
    X_orig = df[list(all_inputs)]
    X = generate_transforms(X_orig)

    st.markdown("## Frequency Analysis (Dominant Frequencies)")
    for c in all_inputs:
        dom_freqs = analyze_fft(df[c].values, sampling_rate)
        st.write(f"Input '{c}': dominant frequencies (Hz): {dom_freqs}")
    for c in io_relations.keys():
        dom_freqs = analyze_fft(df[c].values, sampling_rate)
        st.write(f"Output '{c}': dominant frequencies (Hz): {dom_freqs}")

    for out_col, in_cols_for_output in io_relations.items():
        if len(in_cols_for_output) == 0:
            st.warning(f"No inputs selected for output '{out_col}', skipping.")
            continue
        st.markdown(f"### Modeling Output: **{out_col}** using Inputs: {in_cols_for_output}")

        transform_cols = []
        for in_col in in_cols_for_output:
            transform_cols += [c for c in X.columns if c.startswith(in_col + '_')]
        X_model = X[transform_cols].values
        y = df[out_col].values.reshape(-1, 1)

        for degree in range(1, 4):
            poly = PolynomialFeatures(degree=degree)
            X_poly = poly.fit_transform(X_model)
            lr = LinearRegression()
            lr.fit(X_poly, y)
            y_pred = lr.predict(X_poly)
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            gain = np.sum(lr.coef_[0][1:])
            equation = f"{out_col} = intercept + Σ(coefficients * transformed inputs) [Polynomial degree {degree}]"
            dynamics = estimate_order_and_times(X_model[:, 0], y.flatten(), time_vector, y_pred.flatten())

            results.append({
                'output': out_col,
                'inputs': in_cols_for_output,
                'model': f'Polynomial Degree {degree}',
                'r2': r2,
                'mse': mse,
                'gain': gain,
                'equation': equation,
                'order': dynamics['order'],
                'delay_time': dynamics['delay_time'],
                'process_time': dynamics['process_time'],
                'y_pred': y_pred.flatten()
            })
            st.write(f"Degree {degree}: R2={r2:.4f}, MSE={mse:.4f}, Gain={gain:.4f}")

        # Linear regression on transformed inputs for comparison
        lr = LinearRegression()
        lr.fit(X_model, y)
        y_pred = lr.predict(X_model)
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        gain = np.sum(lr.coef_[0])
        equation = f"{out_col} = intercept + Σ(coefficients * transformed inputs) [Linear]"
        dynamics = estimate_order_and_times(X_model[:, 0], y.flatten(), time_vector, y_pred.flatten())

        results.append({
            'output': out_col,
            'inputs': in_cols_for_output,
            'model': 'Linear',
            'r2': r2,
            'mse': mse,
            'gain': gain,
            'equation': equation,
            'order': dynamics['order'],
            'delay_time': dynamics['delay_time'],
            'process_time': dynamics['process_time'],
            'y_pred': y_pred.flatten()
        })
        st.write(f"Linear: R2={r2:.4f}, MSE={mse:.4f}, Gain={gain:.4f}")

    st.subheader("Summary of All Models")
    df_results = pd.DataFrame(results)
    st.dataframe(df_results[['output', 'inputs', 'model', 'r2', 'mse', 'gain', 'order', 'delay_time', 'process_time', 'equation']])

    for r in results:
        st.write(f"Output: {r['output']} — Model: {r['model']} — Inputs: {r['inputs']}")
        plt.figure(figsize=(10, 4))
        plt.plot(df.index, df[r['output']], label='Actual')
        plt.plot(df.index, r['y_pred'], '--', label='Predicted')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel(r['output'])
        st.pyplot(plt)
        plt.close()

    return results, df

# === Reporting ===
def create_pdf_report(results, df):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Process Analysis Report", 0, 1, 'C')

    pdf.set_font("Arial", size=12)
    for r in results:
        pdf.cell(0, 8, f"Output: {r['output']}", ln=1)
        pdf.cell(0, 8, f"Inputs: {', '.join(r['inputs'])}", ln=1)
        pdf.cell(0, 8, f"Model: {r['model']}", ln=1)
        pdf.cell(0, 8, f"R²: {r['r2']:.4f}, MSE: {r['mse']:.4f}, Gain: {r['gain']:.4f}", ln=1)
        pdf.cell(0, 8, f"Order: {r['order']}, Delay Time: {r['delay_time']:.2f}s, Process Time: {r['process_time']:.2f}s", ln=1)
        pdf.cell(0, 8, f"Equation: {r['equation']}", ln=1)
        pdf.ln(5)

        plt.figure(figsize=(6, 3))
        plt.plot(df.index, df[r['output']], label='Actual')
        plt.plot(df.index, r['y_pred'], '--', label='Predicted')
        plt.title(f"{r['output']} — {r['model']}")
        plt.xlabel('Time')
        plt.ylabel(r['output'])
        plt.legend()
        plt.tight_layout()
        img_path = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
        plt.savefig(img_path)
        plt.close()

        pdf.image(img_path, w=180)
        pdf.ln(10)
    return pdf

def create_excel_report(results, df):
    output = io.BytesIO()
    workbook = xlsxwriter.Workbook(output)
    worksheet = workbook.add_worksheet("Summary")

    headers = ["Output", "Inputs", "Model", "R2", "MSE", "Gain", "Order", "Delay Time (s)", "Process Time (s)", "Equation"]
    for col, header in enumerate(headers):
        worksheet.write(0, col, header)

    for row_num, r in enumerate(results, 1):
        worksheet.write(row_num, 0, r['output'])
        worksheet.write(row_num, 1, ", ".join(r['inputs']))
        worksheet.write(row_num, 2, r['model'])
        worksheet.write(row_num, 3, r['r2'])
        worksheet.write(row_num, 4, r['mse'])
        worksheet.write(row_num, 5, r['gain'])
        worksheet.write(row_num, 6, r['order'])
        worksheet.write(row_num, 7, r['delay_time'])
        worksheet.write(row_num, 8, r['process_time'])
        worksheet.write(row_num, 9, r['equation'])

    for r in results:
        ws = workbook.add_worksheet(f"{r['output']}_{r['model'][:10]}")
        plt.figure(figsize=(8, 4))
        plt.plot(df.index, df[r['output']], label='Actual')
        plt.plot(df.index, r['y_pred'], '--', label='Predicted')
        plt.title(f"{r['output']} — {r['model']}")
        plt.xlabel('Time')
        plt.ylabel(r['output'])
        plt.legend()
        plt.tight_layout()
        imgdata = io.BytesIO()
        plt.savefig(imgdata, format='png')
        plt.close()
        imgdata.seek(0)
        ws.insert_image('B2', f"{r['output']}_plot.png", {'image_data': imgdata, 'x_scale': 0.8, 'y_scale': 0.8})

    workbook.close()
    output.seek(0)
    return output

# === Main app ===
def main():
    st.sidebar.title("Options")
    mode = st.sidebar.radio("Select mode:", ("Upload Data File", "Simulated Real-time Data"))

    if mode == "Upload Data File":
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xls", "xlsx"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                else:
                    st.warning("No 'timestamp' column found. Using default integer index.")

                st.success("File loaded successfully!")

                all_cols = list(df.columns)
                st.sidebar.markdown("### Select Input Columns")
                input_cols = st.sidebar.multiselect("Input columns", options=all_cols, default=all_cols[:1])

                st.sidebar.markdown("### Select Output Columns")
                output_cols = st.sidebar.multiselect("Output columns", options=all_cols, default=all_cols[1:2])

                if len(input_cols) == 0 or len(output_cols) == 0:
                    st.warning("Please select at least one input and one output column.")
                    return

                st.sidebar.markdown("### Define Input-Output Relationships")
                io_relations = {}
                for out_col in output_cols:
                    selected_inputs = st.sidebar.multiselect(
                        f"Select input columns to model for output '{out_col}':",
                        options=input_cols,
                        default=input_cols
                    )
                    io_relations[out_col] = selected_inputs

                if st.button("Run Analysis"):
                    results, df = run_analysis_with_relations(df, io_relations)
                    if results:
                        if st.button("Download Report as Excel"):
                            excel_report = create_excel_report(results, df)
                            st.download_button("Download Excel Report", data=excel_report, file_name="process_report.xlsx")
                        if st.button("Download Report as PDF"):
                            pdf_report = create_pdf_report(results, df)
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                                pdf_report.output(tmp_pdf.name)
                                tmp_pdf.seek(0)
                                pdf_bytes = tmp_pdf.read()
                            st.download_button("Download PDF Report", data=pdf_bytes, file_name="process_report.pdf")

            except Exception as e:
                st.error(f"Error loading file: {e}")

    elif mode == "Simulated Real-time Data":
        st.write("Running simulated real-time streaming data...")
        total_points = st.sidebar.number_input("Total points", min_value=60, max_value=10000, value=1440)
        chunk_size = st.sidebar.number_input("Chunk size", min_value=10, max_value=1000, value=60)

        def simulate_realtime_data(total_points=total_points, chunk_size=chunk_size):
            timestamps = pd.date_range(start='2025-05-01', periods=total_points, freq='T')
            np.random.seed(42)
            input1 = np.linspace(1, 10, total_points)
            input2 = np.sin(np.linspace(0, 20, total_points))
            noise = np.random.normal(0, 0.2, total_points)
            output1 = 2.5 * np.sin(input1) + 1.2 * np.exp(input2) + noise
            output2 = 1.5 * input1 + 0.7 * np.cos(input2) + noise

            df = pd.DataFrame({
                'timestamp': timestamps,
                'input1': input1,
                'input2': input2,
                'output1': output1,
                'output2': output2
            })

            for start in range(0, total_points, chunk_size):
                yield df.iloc[start:start + chunk_size]

        accumulated_df = pd.DataFrame()
        chunk_gen = simulate_realtime_data(total_points, chunk_size)
        for i, chunk in enumerate(chunk_gen, 1):
            st.write(f"Processing chunk {i} of {int(total_points / chunk_size)}...")
            accumulated_df = pd.concat([accumulated_df, chunk]).drop_duplicates(subset='timestamp').reset_index(drop=True)
            accumulated_df['timestamp'] = pd.to_datetime(accumulated_df['timestamp'])
            accumulated_df.set_index('timestamp', inplace=True)
            max_points = 360
            if len(accumulated_df) > max_points:
                accumulated_df = accumulated_df.iloc[-max_points:]

            input_cols = [c for c in accumulated_df.columns if c.lower().startswith('input')]
            output_cols = [c for c in accumulated_df.columns if c.lower().startswith('output')]

            io_relations = {out: input_cols for out in output_cols}

            run_analysis_with_relations(accumulated_df, io_relations)
            time.sleep(1)


if __name__ == "__main__":
    main()