import streamlit as st
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
from sklearn.mixture import GaussianMixture
import string
import random
import sys
import os


def is_discrete(series):
    """Check if a numeric series is likely discrete."""
    return (series.dtype in ['int32', 'int64']) or (series.nunique() / len(series) < 0.05)


def add_noise_to_numeric(series, noise_type="gaussian", scale=1.0, is_discrete_data=False):
    """Add noise to numeric data safely."""
    try:
        series = pd.to_numeric(series, errors='coerce')  # Coerce invalid values to NaN
        if series.isna().any():
            st.warning(f"⚠️ Some values in column were non-numeric and replaced with mean.")
        series = series.fillna(series.mean())  # Replace NaNs with column mean

        if noise_type == "gaussian":
            noise = np.random.normal(0, scale * series.std(), size=len(series))
        else:
            noise = np.random.laplace(0, scale * series.std(), size=len(series))

        result = series + noise
        if is_discrete_data:
            result = np.round(result)
            result = np.maximum(result, series.min())

        return result
    except Exception as e:
        st.error(f"Error in add_noise_to_numeric: {str(e)}")
        return series


def generate_synthetic_numeric(series, is_discrete_data=False):
    """Generate synthetic numeric data preserving statistical properties using GMM."""
    try:
        series = pd.to_numeric(series, errors='coerce')
        data = series.dropna().values.reshape(-1, 1)
        n_components = min(3, len(np.unique(data)))
        gm = GaussianMixture(n_components=n_components, random_state=42)
        gm.fit(data)
        synthetic_data = gm.sample(len(data))[0].flatten()
        if is_discrete_data:
            synthetic_data = np.round(synthetic_data)
            synthetic_data = np.maximum(synthetic_data, series.min())
        return pd.Series(synthetic_data, index=series.dropna().index)
    except Exception as e:
        st.error(f"Error in generate_synthetic_numeric: {str(e)}")
        return series


def infer_column_type(series):
    """Infer the most likely type of data in a column."""
    if series.dtype == 'object':
        try:
            pd.to_datetime(series.iloc[0])
            return "Date"
        except Exception:
            pass

        if series.str.contains('@').any():
            return "Name/ID"
        elif series.str.contains('^EMP\\d+$').any():
            return "Name/ID"
        else:
            return "Categorical"
    elif np.issubdtype(series.dtype, np.number):
        if is_discrete(series):
            return "Numeric (Discrete)"
        else:
            return "Numeric (Continuous)"
    return "Skip"


def generate_random_token(prefix="User", length=3):
    """Generate a random token with a prefix and number."""
    return f"{prefix}{str(length).zfill(3)}"


def anonymize_names(series):
    """Replace names with random tokens while maintaining consistency."""
    try:
        unique_values = series.unique()
        mapping = {value: generate_random_token(length=i+1)
                   for i, value in enumerate(unique_values)}
        return series.map(mapping)
    except Exception as e:
        st.error(f"Error in anonymize_names: {str(e)}")
        return series


def anonymize_categorical(series):
    """Replace categorical values with random codes while preserving frequencies."""
    try:
        unique_values = series.unique()
        mapping = {value: ''.join(random.choices(string.ascii_uppercase, k=5))
                   for value in unique_values}
        return series.map(mapping)
    except Exception as e:
        st.error(f"Error in anonymize_categorical: {str(e)}")
        return series


def anonymize_dates(series, max_shift_days=30):
    """Randomly shift dates within a specified range."""
    try:
        series = pd.to_datetime(series)
        shifts = np.random.randint(-max_shift_days, max_shift_days, size=len(series))
        return series + pd.to_timedelta(shifts, unit='D')
    except Exception as e:
        st.error(f"Error in anonymize_dates: {str(e)}")
        return series


def main():
    st.set_page_config(layout="wide")

    st.title("Data Anonymizer")
    st.write("""
    Upload your Excel or CSV file and select anonymization options for different types of columns.
    The app will create an anonymized version while preserving statistical properties.
    """)

    uploaded_file = st.file_uploader("Choose a file", type=['xlsx', 'csv'])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.write("### Original Data Preview")
            st.dataframe(df.head(10), height=400, use_container_width=True)

            st.sidebar.subheader("Column Configuration")
            column_configs = {}

            for column in df.columns:
                # Styled container for each column
                with st.sidebar.container():
                    st.markdown(
                        f"""
                        <div style="
                            border: 1px solid #444;
                            border-radius: 8px;
                            padding: 12px;
                            margin-bottom: 10px;
                            background-color: #1e1e1e;">
                        <h4 style="margin-top: 0; color: #f5f5f5;">{column}</h4>
                        """,
                        unsafe_allow_html=True
                    )

                    inferred_type = infer_column_type(df[column])
                    st.write(f"Inferred type: {inferred_type}")

                    col_type = st.selectbox(
                        f"Select type for {column}",
                        ["Name/ID", "Numeric (Continuous)", "Numeric (Discrete)", "Categorical", "Date", "Skip"],
                        key=f"type_{column}",
                        index=["Name/ID", "Numeric (Continuous)", "Numeric (Discrete)", "Categorical", "Date", "Skip"].index(inferred_type)
                    )

                    if "Numeric" in col_type:
                        is_discrete = col_type == "Numeric (Discrete)"
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
                                0.01, 1.0, 0.1,
                                key=f"scale_{column}"
                            )
                            column_configs[column] = {
                                "type": col_type,
                                "method": method,
                                "noise_type": noise_type,
                                "scale": scale,
                                "is_discrete": is_discrete
                            }
                        else:
                            column_configs[column] = {
                                "type": col_type,
                                "method": method,
                                "is_discrete": is_discrete
                            }

                    elif col_type == "Date":
                        max_shift = st.slider(
                            "Maximum days to shift",
                            1, 90, 30,
                            key=f"shift_{column}"
                        )
                        column_configs[column] = {
                            "type": col_type,
                            "max_shift": max_shift
                        }
                    else:
                        column_configs[column] = {"type": col_type}

                    # Close div
                    st.markdown("</div>", unsafe_allow_html=True)

            if st.sidebar.button("Anonymize Data", type="primary"):
                with st.spinner("Anonymizing data..."):
                    df_anon = df.copy()

                    for column, config in column_configs.items():
                        try:
                            if config["type"] == "Name/ID":
                                df_anon[column] = anonymize_names(df[column])
                            elif "Numeric" in config["type"]:
                                if config["method"] == "Add Noise":
                                    df_anon[column] = add_noise_to_numeric(
                                        df[column],
                                        config["noise_type"],
                                        config["scale"],
                                        config["is_discrete"]
                                    )
                                else:
                                    df_anon[column] = generate_synthetic_numeric(
                                        df[column],
                                        config["is_discrete"]
                                    )
                            elif config["type"] == "Categorical":
                                df_anon[column] = anonymize_categorical(df[column])
                            elif config["type"] == "Date":
                                df_anon[column] = anonymize_dates(
                                    df[column],
                                    config.get("max_shift", 30)
                                )
                        except Exception as e:
                            st.error(f"Error processing column {column}: {str(e)}")

                    st.write("### Anonymized Data Preview")
                    st.dataframe(df_anon.head(10), height=400, use_container_width=True)

                    try:
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
                    except Exception as e:
                        st.error(f"Error saving file: {str(e)}")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")


if __name__ == "__main__":
    main()
