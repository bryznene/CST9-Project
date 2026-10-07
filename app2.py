import math
import os
import re
from urllib.parse import urlparse
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tldextract

# Page Configuration
st.set_page_config(
    page_title="Phishing URL Detector | CST9",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    /* Card Container */
    .metric-box {
        background-color: #1e222d;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    /* Threat Badges */
    .badge-phish {
        background-color: #ff4b4b;
        color: white;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
    }
    .badge-benign {
        background-color: #00c853;
        color: white;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
    }
    /* Feature Pills */
    .feature-pill {
        background-color: #2b303e;
        color: #e0e0e0;
        padding: 6px 14px;
        border-radius: 15px;
        display: inline-block;
        margin: 4px;
        font-size: 13px;
        border: 1px solid #3d4455;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Load artifacts matching saved pipeline
@st.cache_resource
def load_artifacts():
    model = joblib.load("artifacts/logistic_regression_phishing_model.joblib")
    scaler = joblib.load("artifacts/feature_scaler.joblib")
    feature_cols = joblib.load("artifacts/feature_columns.joblib")

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


# Sidebar
with st.sidebar:
    st.image(
        "https://img.icons8.com/isometric/100/shield-security.png", width=80
    )
    st.title("Phishing URL Detector")
    st.caption("CST9 Final Project | University of Mindanao")
    st.divider()

    st.markdown("### 📊 Model Architecture")
    st.markdown(
        """
    - **Classifier**: Logistic Regression (Config 2)
    - **Dataset Size**: 160,064 sample URLs
    - **Features**: 11 Lexical & Structural Metrics
    - **Target**: Static Content-Independent Detection
    """
    )
    st.divider()

    st.markdown("### ⚠️ Dataset Artifact Notice")
    st.caption(
        "Notice: The training set contains formatting biases (100% of benign URLs lack protocol prefixes and TLD entries). Inputs are processed as-is to reflect true model behavior."
    )


# Header & Title
st.title("🛡️ Phishing URL Detector")
st.markdown(
    "Analyze web links in real-time using **Lexical Feature Extraction** and **Logistic Regression**."
)
st.divider()

# Demo Sample Buttons
st.markdown("**Quick Test Samples:**")
col_s1, col_s2, col_s3 = st.columns(3)
sample_url = ""

if col_s1.button("📌 Phishing Example 1"):
    sample_url = "http://login-verify-account-update.com/signin"
if col_s2.button("📌 Phishing Example 2"):
    sample_url = "http://192.168.1.1/banking/login.php"
if col_s3.button("📌 Dataset Benign Format"):
    sample_url = "google.com"

# Input Box
url_input = st.text_input(
    "Enter or paste URL to analyze:",
    value=sample_url if sample_url else "",
    placeholder="e.g., http://secure-login-portal.com/update",
)

# Execution
if st.button("🔍 Run Security Analysis", type="primary", use_container_width=True):
    if not url_input.strip():
        st.warning("⚠️ Please enter a valid URL to analyze.")
    elif not model_loaded:
        st.error("❌ Model artifacts failed to load properly.")
    else:
        feats = extract_url_features(url_input)
        df_feat = pd.DataFrame([feats])

        # Feature processing
        df_encoded = pd.get_dummies(df_feat)
        df_encoded = df_encoded.reindex(columns=feature_cols, fill_value=0)

        valid_scale_cols = [c for c in scaled_cols if c in df_encoded.columns]
        if valid_scale_cols:
            df_encoded[valid_scale_cols] = scaler.transform(
                df_encoded[valid_scale_cols].to_numpy()
            )

        # Predict
        X_input = df_encoded.to_numpy()
        prediction = model.predict(X_input)[0]
        prob = model.predict_proba(X_input)[0]
        phish_prob = prob[1] * 100

        st.subheader("Analysis Results")

        # Visual Result Summary Card
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            if prediction == 1 or phish_prob > 50.0:
                st.markdown(
                    "<span class='badge-phish'>⚠️ HIGH RISK PHISHING</span>",
                    unsafe_allow_html=True,
                )
                st.metric(
                    label="Phishing Probability", value=f"{phish_prob:.2f}%"
                )
            else:
                st.markdown(
                    "<span class='badge-benign'>✅ LOW RISK BENIGN</span>",
                    unsafe_allow_html=True,
                )
                st.metric(
                    label="Legitimate Probability",
                    value=f"{100 - phish_prob:.2f}%",
                )

        with res_col2:
            st.write("**Risk Meter:**")
            st.progress(int(phish_prob))
            if phish_prob > 50.0:
                st.caption(
                    "🚨 This URL exhibits lexical patterns frequently observed in phishing vectors."
                )
            else:
                st.caption(
                    "✅ This URL exhibits lexical structure aligning with benign samples."
                )

        st.divider()

        # Detailed Tabs
        tab1, tab2 = st.tabs(
            ["📊 Extracted Feature Metrics", "💻 Raw JSON Breakdown"]
        )

        with tab1:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("URL Length", feats["url_length"])
            m2.metric("Dot Count", feats["num_dots"])
            m3.metric("HTTPS Enabled", "Yes" if feats["has_https"] else "No")
            m4.metric("IP Domain", "Yes" if feats["has_ip"] else "No")

            m5, m6, m7, m8 = st.columns(4)
            m5.metric("Subdirectories", feats["num_subdirs"])
            m6.metric("Parameters", feats["num_params"])
            m7.metric("Suspicious Words", feats["suspicious_words"])
            m8.metric("TLD Suffix", feats["tld"])

            st.write("**Structural Entropy & Character Counts:**")
            st.markdown(
                f"""
                <div style="background-color: #1e222d; padding: 15px; border-radius: 8px;">
                    <span class="feature-pill"><b>Entropy:</b> {feats['entropy']:.4f}</span>
                    <span class="feature-pill"><b>Special Chars:</b> {feats['special_char_count']}</span>
                    <span class="feature-pill"><b>Digits Count:</b> {feats['digits_count']}</span>
                </div>
            """,
                unsafe_allow_html=True,
            )

        with tab2:
            st.json(feats)
