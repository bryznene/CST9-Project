import math
import os
import re
from urllib.parse import urlparse
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tldextract

# Page Configuration & Layout
st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded",
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
    st.error(f"Error loading model artifacts: {e}")


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
    found_keywords = [kw for kw in keywords if kw in url.lower()]
    suspicious_words = len(found_keywords)

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
        "found_keywords": found_keywords,
        "tld": tld,
        "special_char_count": special_char_count,
        "digits_count": digits_count,
        "entropy": entropy,
    }


def generate_threat_reasons(feats):
    reasons = []
    if feats["has_ip"] == 1:
        reasons.append(
            "Uses a direct IP address instead of a standard domain name."
        )
    if feats["suspicious_words"] > 0:
        words_str = ", ".join([f"'{w}'" for w in feats["found_keywords"]])
        reasons.append(
            f"Contains sensitive target keyword(s): {words_str}."
        )
    if feats["url_length"] > 75:
        reasons.append(
            f"Excessively long URL string ({feats['url_length']} characters)."
        )
    if feats["num_dots"] > 3:
        reasons.append(
            f"High sub-domain depth detected ({feats['num_dots']} dots)."
        )
    if feats["entropy"] > 4.2:
        reasons.append(
            f"High character entropy ({feats['entropy']:.2f}) suggesting random string obfuscation."
        )
    if feats["special_char_count"] > 10:
        reasons.append(
            f"Unusual density of special characters ({feats['special_char_count']} special symbols)."
        )

    if not reasons:
        reasons.append(
            "Overall structural features resemble typical phishing pattern distribution in training data."
        )

    return reasons


# Sidebar Info
with st.sidebar:
    st.header("Model Info")
    st.info(
        """
    - **Model**: Logistic Regression (Config 2)
    - **Scope**: Lexical Feature Analysis
    - **Dataset**: 160,064 records
    """
    )
    st.divider()
    st.caption("CST9 Project | University of Mindanao")


# UI Header
st.title("🛡️ Phishing URL Detector")
st.markdown(
    "Analyze web links in real-time using **Lexical Feature Extraction** and **Logistic Regression**."
)
st.divider()

# Input Form
url_input = st.text_input(
    "Enter URL to Analyze:",
    placeholder="e.g., http://login-verify-account.com/signin",
)

if st.button("Analyze URL", type="primary", use_container_width=True):
    if not url_input.strip():
        st.warning("Please enter a URL to inspect.")
    elif not model_loaded:
        st.error("Model artifacts failed to load.")
    else:
        feats = extract_url_features(url_input)

        # Prepare DataFrame for pipeline
        feat_dict_for_df = feats.copy()
        feat_dict_for_df.pop("found_keywords", None)
        df_feat = pd.DataFrame([feat_dict_for_df])

        # Encode and Scale
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

        st.divider()

        # Result Banner & Metrics
        col_res1, col_res2 = st.columns([1, 1])

        with col_res1:
            if prediction == 1 or phish_prob > 50.0:
                st.error("**HIGH RISK: PHISHING DETECTED**")
                st.metric("Phishing Probability", f"{phish_prob:.2f}%")
            else:
                st.success("**LOW RISK: BENIGN URL**")
                st.metric("Legitimate Confidence", f"{100 - phish_prob:.2f}%")

        with col_res2:
            st.write("**Threat Confidence Level:**")
            st.progress(int(phish_prob) / 100)

            if phish_prob > 50.0:
                st.caption(
                    "**Recommendation**: Do not enter credentials or download files from this link."
                )
            else:
                st.caption(
                    "**Recommendation**: URL structure aligns with standard benign patterns."
                )

        st.divider()

        # Suspicious Reason Analysis
        if prediction == 1 or phish_prob > 50.0:
            st.subheader("Why is this URL suspicious?")
            reasons = generate_threat_reasons(feats)
            for r in reasons:
                st.markdown(f"-{r}")
            st.divider()

        # Clean Feature Table (Replaces raw JSON)
        st.subheader("Extracted Lexical Features")

        table_data = {
            "Feature Metric": [
                "URL Length",
                "Dot Count",
                "HTTPS Protocol",
                "IP Address Usage",
                "Subdirectory Depth",
                "Query Parameters",
                "Suspicious Keywords",
                "TLD Suffix",
                "Special Character Count",
                "Digits Count",
                "Shannon Entropy Score",
            ],
            "Value": [
                f"{feats['url_length']} chars",
                feats["num_dots"],
                "Yes" if feats["has_https"] else "No",
                "Yes" if feats["has_ip"] else "No",
                feats["num_subdirs"],
                feats["num_params"],
                feats["suspicious_words"],
                feats["tld"],
                feats["special_char_count"],
                feats["digits_count"],
                f"{feats['entropy']:.4f}",
            ],
        }

        st.table(pd.DataFrame(table_data))
