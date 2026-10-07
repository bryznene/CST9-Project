import math
import os
import re
from urllib.parse import urlparse
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tldextract

st.set_page_config(
    page_title="Phishing URL Detector", page_icon="🛡️", layout="centered"
)


# Load artifacts matching your saved pipeline
@st.cache_resource
def load_artifacts():
    model = joblib.load("artifacts/logistic_regression_phishing_model.joblib")
    scaler = joblib.load("artifacts/feature_scaler.joblib")
    feature_cols = joblib.load("artifacts/feature_columns.joblib")

    # Load scaled columns if present, otherwise default to numeric feature list
    scaled_cols_path = "artifacts/scaled_columns.joblib"
    if os.path.exists(scaled_cols_path):
        scaled_cols = joblib.load(scaled_cols_path)
    else:
        scaled_cols = [
            "url_length",
            "num_dots",
            "has_https",
            "has_ip",
            "num_subdirs",
            "num_params",
            "suspicious_words",
            "special_char_count",
            "digits_count",
            "entropy",
        ]

    return model, scaler, feature_cols, scaled_cols


try:
    model, scaler, feature_cols, scaled_cols = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading artifacts: {e}")


def calculate_entropy(text):
    if not text:
        return 0.0
    prob = [float(text.count(c)) / len(text) for c in set(text)]
    return -sum([p * math.log(p, 2) for p in prob])


def extract_url_features(url):
    parsed = urlparse(url)
    ext = tldextract.extract(url)

    url_length = len(url)
    num_dots = url.count(".")
    has_https = 1 if parsed.scheme.lower() == "https" else 0

    domain = parsed.netloc or parsed.path.split("/")[0]
    ip_pattern = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
    has_ip = 1 if ip_pattern.match(domain.split(":")[0]) else 0

    path = parsed.path
    num_subdirs = max(
        0, path.count("/") - (1 if path.endswith("/") else 0)
    )
    num_params = len(parsed.query.split("&")) if parsed.query else 0

    keywords = [
        "login",
        "verify",
        "update",
        "secure",
        "account",
        "banking",
        "signin",
        "confirm",
    ]
    suspicious_words = sum(1 for kw in keywords if kw in url.lower())

    special_chars = r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,<>\/?]"
    special_char_count = len(re.findall(special_chars, url))
    digits_count = sum(c.isdigit() for c in url)
    entropy = calculate_entropy(url)

    tld = ext.suffix if ext.suffix else "missing"

    return {
        "url_length": url_length,
        "num_dots": num_dots,
        "has_https": has_https,
        "has_ip": has_ip,
        "num_subdirs": num_subdirs,
        "num_params": num_params,
        "suspicious_words": suspicious_words,
        "tld": tld,
        "special_char_count": special_char_count,
        "digits_count": digits_count,
        "entropy": entropy,
    }


# UI
st.title("🛡️ Phishing URL Detection System")
st.markdown("Enter a URL to analyze its lexical features using Logistic Regression.")

url_input = st.text_input(
    "Enter URL:", placeholder="http://182.116.11.31:57515/bin.sh"
)

if st.button("Analyze URL", type="primary"):
    if not url_input.strip():
        st.warning("Please enter a valid URL.")
    elif not model_loaded:
        st.error("Model artifacts not loaded properly.")
    else:
        feats = extract_url_features(url_input)
        df_feat = pd.DataFrame([feats])

        # One-hot encode and reindex to match model feature columns
        df_encoded = pd.get_dummies(df_feat)
        df_encoded = df_encoded.reindex(columns=feature_cols, fill_value=0)

        # Scale only the numeric columns that were originally scaled
        valid_scale_cols = [c for c in scaled_cols if c in df_encoded.columns]
        if valid_scale_cols:
            # Convert to numpy array to strip feature names and prevent scikit-learn mismatch errors
            df_encoded[valid_scale_cols] = scaler.transform(
                df_encoded[valid_scale_cols].to_numpy()
            )

        # Predict using numpy values
        X_input = df_encoded.to_numpy()
        prediction = model.predict(X_input)[0]
        prob = model.predict_proba(X_input)[0]
        phish_prob = prob[1] * 100

        st.divider()

        # 1. Result Status Card
        if prediction == 1 or phish_prob > 50.0:
            st.error(
                "### HIGH RISK: PHISHING URL DETECTED\n\n"
                "The machine learning model classified this URL as potentially malicious. "
                "Avoid entering personal or sensitive information on this website."
            )
        else:
            st.success(
                "### LOW RISK: BENIGN URL DETECTED\n\n"
                "The machine learning model classified this URL as low risk based on its lexical features. "
                "Always verify the destination domain before submitting credentials."
            )

        # 2. Probability & Progress Bar
        st.markdown("**Phishing Probability**")
        st.markdown(f"## {phish_prob:.2f}%")
        st.progress(float(phish_prob) / 100.0)

        st.write("")

        # 3. Analyzed URL Display
        st.markdown("**Analyzed URL**")
        st.code(url_input, language=None)

        st.write("")

        # 4. Extracted URL Features Table
        st.markdown("### Extracted URL Features")

        feature_table_data = {
            "Feature": [
                "URL Length",
                "Number of Dots",
                "HTTPS",
                "IP Address",
                "Number of Subdirectories",
                "Number of Parameters",
                "Suspicious Keywords",
                "Top-Level Domain",
                "Special Characters",
                "Digits",
            ],
            "Value": [
                str(feats["url_length"]),
                str(feats["num_dots"]),
                "Yes" if feats["has_https"] == 1 else "No",
                "Yes" if feats["has_ip"] == 1 else "No",
                str(feats["num_subdirs"]),
