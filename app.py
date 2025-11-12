import streamlit as st
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
from sklearn.mixture import GaussianMixture
from sdv.tabular import CTGAN
import string
import random

def generate_random_token(prefix="User", length=3):
    """Generate a random token with a prefix and number."""
    return f"{prefix}{str(length).zfill(3)}"

def anonymize_names(series):
    """Replace names with random tokens while maintaining consistency."""
    unique_values = series.unique()
    mapping = {value: generate_random_token(length=i+1) 
              for i, value in enumerate(unique_values)}
    return series.map(mapping)

def add_noise_to_numeric(series, noise_type="gaussian", scale=1.0):
    """Add noise to numeric data."""
    if noise_type == "gaussian":
        noise = np.random.normal(0, scale, size=len(series))
    else:  # Laplace noise
        noise = np.random.laplace(0, scale, size=len(series))
    return series + noise

def generate_synthetic_numeric(series, method="gaussian_mixture"):
    """Generate synthetic numeric data preserving statistical properties."""
    data = series.values.reshape(-1, 1)
    if method == "gaussian_mixture":
        gm = GaussianMixture(n_components=3, random_state=42)
        gm.fit(data)
        synthetic_data = gm.sample(len(data))[0].flatten()
    else:  # CTGAN
        ctgan = CTGAN()
        df_temp = pd.DataFrame({series.name: series})
        ctgan.fit(df_temp)
        synthetic_data = ctgan.sample(len(series))[series.name]
    return pd.Series(synthetic_data)

def anonymize_categorical(series):
    """Replace categorical values with random codes while preserving frequencies."""
    unique_values = series.unique()
    mapping = {value: ''.join(random.choices(string.ascii_uppercase, k=5)) 
              for value in unique_values}
    return series.map(mapping)

def anonymize_dates(series, max_shift_days=30):
    """Randomly shift dates within a specified range."""
    shifts = np.random.randint(-max_shift_days, max_shift_days, size=len(series))
    return series.apply(lambda x: x + timedelta(days=int(shifts[series.index.get_loc(x)])))

def main():
    st.title("Data Anonymizer")
    st.write("""
    Upload your Excel or CSV file and select anonymization options for different types of columns.
    The app will create an anonymized version while preserving statistical properties.
    """)

    uploaded_file = st.file_uploader("Choose a file", type=['xlsx', 'csv'])
    
    if uploaded_file is not None:
        # Read the file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.write("Original Data Preview:")
        st.dataframe(df.head())
        
        # Column type selection
        st.subheader("Column Configuration")
        column_configs = {}
        
        for column in df.columns:
            st.write(f"### {column}")
            col_type = st.selectbox(
                f"Select type for {column}",
                ["Name/ID", "Numeric", "Categorical", "Date", "Skip"],
                key=f"type_{column}"
            )
            
            if col_type == "Numeric":
                method = st.selectbox(
                    f"Select anonymization method for {column}",
                    ["Add Noise", "Synthetic Data"],
                    key=f"method_{column}"
                )
                if method == "Add Noise":
                    noise_type = st.selectbox(
                        "Noise distribution",
                        ["gaussian", "laplace"],
                        key=f"noise_type_{column}"
                    )
                    scale = st.slider(
                        "Noise scale",
                        0.1, 10.0, 1.0,
                        key=f"scale_{column}"
                    )
                    column_configs[column] = {
                        "type": col_type,
                        "method": method,
                        "noise_type": noise_type,
                        "scale": scale
                    }
                else:
                    synth_method = st.selectbox(
                        "Synthetic data method",
                        ["gaussian_mixture", "ctgan"],
                        key=f"synth_method_{column}"
                    )
                    column_configs[column] = {
                        "type": col_type,
                        "method": method,
                        "synth_method": synth_method
                    }
            else:
                column_configs[column] = {"type": col_type}
        
        if st.button("Anonymize Data"):
            df_anon = df.copy()
            
            for column, config in column_configs.items():
                if config["type"] == "Name/ID":
                    df_anon[column] = anonymize_names(df[column])
                elif config["type"] == "Numeric":
                    if config["method"] == "Add Noise":
                        df_anon[column] = add_noise_to_numeric(
                            df[column],
                            config["noise_type"],
                            config["scale"]
                        )
                    else:  # Synthetic Data
                        df_anon[column] = generate_synthetic_numeric(
                            df[column],
                            config["synth_method"]
                        )
                elif config["type"] == "Categorical":
                    df_anon[column] = anonymize_categorical(df[column])
                elif config["type"] == "Date":
                    df_anon[column] = anonymize_dates(pd.to_datetime(df[column]))
            
            st.write("Anonymized Data Preview:")
            st.dataframe(df_anon.head())
            
            # Save to Excel
            output = pd.ExcelWriter('anonymized_data.xlsx', engine='openpyxl')
            df_anon.to_excel(output, index=False)
            output.close()
            
            with open('anonymized_data.xlsx', 'rb') as f:
                st.download_button(
                    label="Download Anonymized Data",
                    data=f,
                    file_name="anonymized_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

if __name__ == "__main__":
    main() 