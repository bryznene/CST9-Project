import math
import os
import re
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tldextract


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Phishing URL Detection System",
    page_icon="🛡️",
    layout="centered"
)


# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

    /* Main page */
    .main {
        padding-top: 2rem;
    }

    /* Main title */
    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 16px;
        color: #a0a0a0;
        margin-bottom: 30px;
    }

    /* Result cards */
    .result-card {
        padding: 25px;
        border-radius: 12px;
        margin-top: 20px;
        margin-bottom: 20px;
        border: 1px solid rgba(255,255,255,0.1);
    }

    .danger-card {
        background-color: rgba(220, 53, 69, 0.12);
        border: 1px solid rgba(220, 53, 69, 0.45);
    }

    .safe-card {
        background-color: rgba(25, 135, 84, 0.12);
        border: 1px solid rgba(25, 135, 84, 0.45);
    }

    .result-title {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .result-description {
        font-size: 14px;
        color: #bdbdbd;
    }

    /* Probability */
    .probability-label {
        font-size: 15px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 5px;
    }

    .probability-value {
        font-size: 36px;
        font-weight: 700;
    }

    /* Feature section */
    .section-title {
        font-size: 21px;
        font-weight: 600;
        margin-top: 25px;
        margin-bottom: 10px;
    }

    /* Info cards */
    .info-card {
        padding: 18px;
        border-radius: 10px;
        background-color: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 10px;
    }

    .info-label {
        font-size: 13px;
        color: #999999;
    }

    .info-value {
        font-size: 18px;
        font-weight: 600;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #777777;
        font-size: 13px;
        margin-top: 45px;
        padding-bottom: 20px;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL ARTIFACTS
# =========================================================
@st.cache_resource
def load_artifacts():

    model = joblib.load(
        "artifacts/logistic_regression_phishing_model.joblib"
    )

    scaler = joblib.load(
        "artifacts/feature_scaler.joblib"
    )

    feature_cols = joblib.load(
        "artifacts/feature_columns.joblib"
    )

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


# =========================================================
# LOAD MODEL
# =========================================================
try:

    model, scaler, feature_cols, scaled_cols = load_artifacts()

    model_loaded = True

except Exception as e:

    model_loaded = False

    st.error(
        f"Unable to load the model artifacts. "
        f"Please check the artifacts folder.\n\n{e}"
    )


# =========================================================
# ENTROPY CALCULATION
# =========================================================
def calculate_entropy(text):

    if not text:
        return 0.0

    prob = [
        float(text.count(c)) / len(text)
        for c in set(text)
    ]

    return -sum(
        p * math.log(p, 2)
        for p in prob
    )


# =========================================================
# URL FEATURE EXTRACTION
# =========================================================
def extract_url_features(url):

    parsed = urlparse(url)

    ext = tldextract.extract(url)

    # URL length
    url_length = len(url)

    # Number of dots
    num_dots = url.count(".")

    # HTTPS
    has_https = (
        1
        if parsed.scheme.lower() == "https"
        else 0
    )

    # IP address detection
    domain = (
        parsed.netloc
        or parsed.path.split("/")[0]
    )

    ip_pattern = re.compile(
        r"^\d{1,3}\."
        r"\d{1,3}\."
        r"\d{1,3}\."
        r"\d{1,3}$"
    )

    has_ip = (
        1
        if ip_pattern.match(
            domain.split(":")[0]
        )
        else 0
    )

    # Path depth
    path = parsed.path

    num_subdirs = max(
        0,
        path.count("/")
        - (1 if path.endswith("/") else 0)
    )

    # URL parameters
    num_params = (
        len(parsed.query.split("&"))
        if parsed.query
        else 0
    )

    # Suspicious keywords
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

    suspicious_words = sum(
        1
        for kw in keywords
        if kw in url.lower()
    )

    # Special characters
    special_chars = (
        r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,<>\/?]"
    )

    special_char_count = len(
        re.findall(
            special_chars,
            url
        )
    )

    # Digits
    digits_count = sum(
        c.isdigit()
        for c in url
    )

    # Entropy
    entropy = calculate_entropy(url)

    # TLD
    tld = (
        ext.suffix
        if ext.suffix
        else "missing"
    )

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


# =========================================================
# HEADER
# =========================================================
st.markdown(
    '<div class="main-title">Phishing URL Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine learning-based detection of potentially malicious URLs '
    'using lexical URL features.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# URL INPUT
# =========================================================
st.markdown("### URL Analysis")

url_input = st.text_input(
    "Enter URL",
    placeholder="https://example.com/login",
    label_visibility="collapsed"
)

st.caption(
    "Enter the complete URL you want the system to analyze."
)


# =========================================================
# ANALYZE BUTTON
# =========================================================
analyze = st.button(
    "Analyze URL",
    type="primary",
    use_container_width=True
)


# =========================================================
# ANALYSIS
# =========================================================
if analyze:

    # Empty URL
    if not url_input.strip():

        st.warning(
            "Please enter a URL before starting the analysis."
        )

    # Model unavailable
    elif not model_loaded:

        st.error(
            "The machine learning model could not be loaded."
        )

    else:

        # -------------------------------------------------
        # Basic URL validation
        # -------------------------------------------------
        clean_url = url_input.strip()

        if not re.match(
            r"^https?://",
            clean_url,
            re.IGNORECASE
        ):

            st.warning(
                "Please enter a complete URL beginning "
                "with http:// or https://."
            )

        else:

            # -------------------------------------------------
            # Extract features
            # -------------------------------------------------
            feats = extract_url_features(
                clean_url
            )

            df_feat = pd.DataFrame(
                [feats]
            )

            # -------------------------------------------------
            # Encode TLD
            # -------------------------------------------------
            df_encoded = pd.get_dummies(
                df_feat
            )

            # Match model columns
            df_encoded = df_encoded.reindex(
                columns=feature_cols,
                fill_value=0
            )

            # -------------------------------------------------
            # Scale numeric columns
            # -------------------------------------------------
            valid_scale_cols = [
                c
                for c in scaled_cols
                if c in df_encoded.columns
            ]

            if valid_scale_cols:

                df_encoded[
                    valid_scale_cols
                ] = scaler.transform(
                    df_encoded[
                        valid_scale_cols
                    ].to_numpy()
                )

            # -------------------------------------------------
            # Prediction
            # -------------------------------------------------
            X_input = df_encoded.to_numpy()

            prediction = model.predict(
                X_input
            )[0]

            prob = model.predict_proba(
                X_input
            )[0]

            phish_prob = prob[1] * 100

            legitimate_prob = (
                100 - phish_prob
            )

            # =================================================
            # RESULT
            # =================================================
            st.divider()

            if prediction == 1 or phish_prob > 50:

                # -------------------------------
                # PHISHING RESULT
                # -------------------------------
                st.markdown(
                    """
                    <div class="result-card danger-card">
                        <div class="result-title">
                            HIGH RISK: PHISHING URL DETECTED
                        </div>
                        <div class="result-description">
                            The machine learning model classified
                            this URL as potentially malicious.
                            Avoid entering personal or sensitive
                            information on this website.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="probability-label">'
                    'Phishing Probability'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="probability-value">'
                    f'{phish_prob:.2f}%'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.progress(
                    min(phish_prob / 100, 1.0)
                )

            else:

                # -------------------------------
                # BENIGN RESULT
                # -------------------------------
                st.markdown(
                    """
                    <div class="result-card safe-card">
                        <div class="result-title">
                            LOW RISK: BENIGN URL
                        </div>
                        <div class="result-description">
                            The machine learning model classified
                            this URL as likely legitimate.
                            However, users should still exercise
                            caution when visiting unfamiliar websites.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="probability-label">'
                    'Legitimate Probability'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="probability-value">'
                    f'{legitimate_prob:.2f}%'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.progress(
                    min(legitimate_prob / 100, 1.0)
                )

            # =================================================
            # URL INFORMATION
            # =================================================
            st.markdown(
                '<div class="section-title">'
                'Analyzed URL'
                '</div>',
                unsafe_allow_html=True
            )

            st.code(
                clean_url,
                language=None
            )

            # =================================================
            # FEATURE ANALYSIS
            # =================================================
            st.markdown(
                '<div class="section-title">'
                'Extracted URL Features'
                '</div>',
                unsafe_allow_html=True
            )

            # Create readable feature table
            feature_display = pd.DataFrame({
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
                    "Entropy",
                ],

                "Value": [
                    feats["url_length"],
                    feats["num_dots"],
                    "Yes"
                    if feats["has_https"]
                    else "No",
                    "Yes"
                    if feats["has_ip"]
                    else "No",
                    feats["num_subdirs"],
                    feats["num_params"],
                    feats["suspicious_words"],
                    feats["tld"],
                    feats["special_char_count"],
                    feats["digits_count"],
                    f'{feats["entropy"]:.4f}',
                ]
            })

            st.dataframe(
                feature_display,
                use_container_width=True,
                hide_index=True
            )

            # =================================================
            # MODEL INFORMATION
            # =================================================
            st.markdown(
                '<div class="section-title">'
                'Detection Model'
                '</div>',
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    """
                    <div class="info-card">
                        <div class="info-label">
                            Machine Learning Algorithm
                        </div>
                        <div class="info-value">
                            Logistic Regression
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:

                st.markdown(
                    """
                    <div class="info-card">
                        <div class="info-label">
                            Analysis Type
                        </div>
                        <div class="info-value">
                            Lexical URL Analysis
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # =================================================
            # DISCLAIMER
            # =================================================
            st.info(
                "Detection results are predictions generated by "
                "a machine learning model. A high-risk result "
                "does not guarantee that a website is malicious, "
                "and a low-risk result does not guarantee complete safety."
            )


# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    <div class="footer">
        Phishing URL Detection System<br>
        Machine Learning-Based URL Classification
    </div>
    """,
    unsafe_allow_html=True
)
