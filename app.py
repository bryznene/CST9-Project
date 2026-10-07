import math
import os
import re
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tldextract


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Phishing URL Detection System",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- Global ---------- */

    .stApp {
        background: #0d1117;
    }

    .block-container {
        max-width: 980px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #f0f6fc !important;
    }

    p, label {
        color: #c9d1d9;
    }

    /* ---------- Header ---------- */

    .hero {
        margin-bottom: 2rem;
    }

    .hero-title {
        font-size: 2.45rem;
        font-weight: 800;
        color: #f0f6fc;
        letter-spacing: -0.8px;
        line-height: 1.15;
        margin-bottom: 0.45rem;
    }

    .hero-subtitle {
        color: #8b949e;
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 0;
    }

    .shield {
        color: #58a6ff;
    }

    /* ---------- Input ---------- */

    div[data-testid="stTextInput"] input {
        background: #161b22;
        color: #f0f6fc;
        border: 1px solid #30363d;
        border-radius: 8px;
        height: 46px;
        font-size: 0.95rem;
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: #58a6ff;
        box-shadow: 0 0 0 1px #58a6ff;
    }

    /* ---------- Buttons ---------- */

    .stButton > button {
        border-radius: 8px;
        min-height: 42px;
        font-weight: 700;
        border: 1px solid #30363d;
    }

    /* ---------- Result Cards ---------- */

    .risk-card {
        padding: 1.35rem 1.45rem;
        border-radius: 10px;
        margin-top: 1.4rem;
        margin-bottom: 1.5rem;
        border: 1px solid;
    }

    .risk-low {
        background: linear-gradient(
            135deg,
            rgba(35, 134, 54, 0.25),
            rgba(35, 134, 54, 0.10)
        );
        border-color: #238636;
    }

    .risk-high {
        background: linear-gradient(
            135deg,
            rgba(248, 81, 73, 0.24),
            rgba(248, 81, 73, 0.08)
        );
        border-color: #f85149;
    }

    .risk-title {
        font-size: 1.35rem;
        font-weight: 800;
        margin-bottom: 0.45rem;
    }

    .risk-low .risk-title {
        color: #3fb950;
    }

    .risk-high .risk-title {
        color: #ff7b72;
    }

    .risk-description {
        color: #c9d1d9;
        font-size: 0.92rem;
        line-height: 1.55;
    }

    /* ---------- Probability ---------- */

    .probability-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1.4rem;
        margin-bottom: 1.5rem;
    }

    .probability-label {
        color: #8b949e;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }

    .probability-number {
        color: #f0f6fc;
        font-size: 2.5rem;
        font-weight: 800;
        margin-top: 0.25rem;
    }

    .probability-description {
        color: #8b949e;
        font-size: 0.85rem;
    }

    .probability-bar {
        height: 9px;
        background: #21262d;
        border-radius: 99px;
        overflow: hidden;
        margin-top: 1rem;
    }

    .probability-fill {
        height: 100%;
        border-radius: 99px;
    }

    /* ---------- URL Card ---------- */

    .url-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1rem 1.1rem;
        margin-bottom: 1.8rem;
    }

    .url-label {
        color: #8b949e;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        margin-bottom: 0.45rem;
    }

    .url-value {
        color: #58a6ff;
        font-family: monospace;
        font-size: 0.92rem;
        word-break: break-all;
    }

    /* ---------- Feature Cards ---------- */

    .feature-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 9px;
        padding: 1rem;
        min-height: 88px;
    }

    .feature-label {
        color: #8b949e;
        font-size: 0.75rem;
        margin-bottom: 0.35rem;
    }

    .feature-value {
        color: #f0f6fc;
        font-size: 1.15rem;
        font-weight: 750;
    }

    /* ---------- Model Cards ---------- */

    .model-card {
        background: linear-gradient(
            135deg,
            rgba(31, 111, 235, 0.20),
            rgba(31, 111, 235, 0.08)
        );
        border: 1px solid #1f6feb;
        border-radius: 10px;
        padding: 1.1rem;
        min-height: 105px;
    }

    .model-label {
        color: #58a6ff;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .model-value {
        color: #f0f6fc;
        font-size: 1.15rem;
        font-weight: 750;
    }

    /* ---------- Disclaimer ---------- */

    .disclaimer {
        background: #161b22;
        border: 1px solid #30363d;
        border-left: 4px solid #58a6ff;
        border-radius: 7px;
        padding: 0.9rem 1rem;
        margin-top: 1.5rem;
        color: #8b949e;
        font-size: 0.8rem;
        line-height: 1.55;
    }

    /* ---------- Section divider ---------- */

    .section-title {
        margin-top: 1.8rem;
        margin-bottom: 0.8rem;
        font-size: 1.25rem;
        font-weight: 750;
        color: #f0f6fc;
    }

    .small-note {
        color: #8b949e;
        font-size: 0.8rem;
        margin-top: -0.35rem;
        margin-bottom: 1rem;
    }

    /* ---------- Dataframe ---------- */

    div[data-testid="stDataFrame"] {
        border: 1px solid #30363d;
        border-radius: 8px;
        overflow: hidden;
    }

    /* ---------- Footer ---------- */

    .footer {
        text-align: center;
        color: #484f58;
        font-size: 0.75rem;
        margin-top: 2.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

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


try:
    model, scaler, feature_cols, scaled_cols = load_artifacts()
    model_loaded = True
    model_error = None

except Exception as e:
    model_loaded = False
    model_error = str(e)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def calculate_entropy(text):
    if not text:
        return 0.0

    probabilities = [
        float(text.count(char)) / len(text)
        for char in set(text)
    ]

    return -sum(
        p * math.log(p, 2)
        for p in probabilities
    )


def normalize_url(url):
    """
    Adds HTTPS when the user enters a domain without a scheme.
    """

    url = url.strip()

    if not url:
        return ""

    parsed = urlparse(url)

    if not parsed.scheme:
        url = "https://" + url

    return url


def extract_url_features(url):

    parsed = urlparse(url)
    ext = tldextract.extract(url)

    url_length = len(url)

    num_dots = url.count(".")

    has_https = (
        1
        if parsed.scheme.lower() == "https"
        else 0
    )

    domain = parsed.netloc or parsed.path.split("/")[0]

    ip_pattern = re.compile(
        r"^\d{1,3}\."
        r"\d{1,3}\."
        r"\d{1,3}\."
        r"\d{1,3}$"
    )

    has_ip = (
        1
        if ip_pattern.match(domain.split(":")[0])
        else 0
    )

    path = parsed.path

    num_subdirs = max(
        0,
        path.count("/")
        - (1 if path.endswith("/") else 0)
    )

    num_params = (
        len(parsed.query.split("&"))
        if parsed.query
        else 0
    )

    suspicious_keywords = [
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
        for keyword in suspicious_keywords
        if keyword in url.lower()
    )

    special_chars_pattern = (
        r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,<>\/?]"
    )

    special_char_count = len(
        re.findall(
            special_chars_pattern,
            url
        )
    )

    digits_count = sum(
        char.isdigit()
        for char in url
    )

    entropy = calculate_entropy(url)

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


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            <span class="shield">🛡️</span>
            Phishing URL Detection System
        </div>

        <div class="hero-subtitle">
            Machine learning-based detection of potentially
            malicious URLs using lexical URL features.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL ERROR
# ============================================================

if not model_loaded:

    st.error(
        f"Unable to load the detection model.\n\n"
        f"Error: {model_error}"
    )

    st.stop()


# ============================================================
# URL INPUT
# ============================================================

st.markdown("### Analyze a URL")

url_input = st.text_input(
    "Enter URL",
    placeholder="https://example.com/login",
    label_visibility="collapsed",
)


# Example buttons

st.markdown(
    '<div class="small-note">Try an example:</div>',
    unsafe_allow_html=True,
)

example_col1, example_col2, example_col3 = st.columns(3)

with example_col1:
    if st.button(
        "🌐 Example Domain",
        use_container_width=True,
    ):
        url_input = "https://example.com"

with example_col2:
    if st.button(
        "🔐 Login Example",
        use_container_width=True,
    ):
        url_input = "https://example.com/login"

with example_col3:
    if st.button(
        "⚠️ Suspicious Example",
        use_container_width=True,
    ):
        url_input = (
            "http://192.168.1.10/"
            "verify/account/login.php?"
            "session=123456"
        )


analyze = st.button(
    "🔍 Analyze URL",
    type="primary",
    use_container_width=True,
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze:

    if not url_input.strip():

        st.warning(
            "Please enter a URL before starting the analysis."
        )

        st.stop()

    normalized_url = normalize_url(url_input)

    try:

        # ----------------------------------------------------
        # Extract features
        # ----------------------------------------------------

        feats = extract_url_features(
            normalized_url
        )

        df_feat = pd.DataFrame([feats])

        # ----------------------------------------------------
        # One-hot encode
        # ----------------------------------------------------

        df_encoded = pd.get_dummies(
            df_feat
        )

        df_encoded = df_encoded.reindex(
            columns=feature_cols,
            fill_value=0,
        )

        # ----------------------------------------------------
        # Scale numerical features
        # ----------------------------------------------------

        valid_scale_cols = [
            column
            for column in scaled_cols
            if column in df_encoded.columns
        ]

        if valid_scale_cols:

            df_encoded[valid_scale_cols] = (
                scaler.transform(
                    df_encoded[
                        valid_scale_cols
                    ].to_numpy()
                )
            )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        X_input = df_encoded.to_numpy()

        prediction = model.predict(
            X_input
        )[0]

        probabilities = model.predict_proba(
            X_input
        )[0]

        # Safely identify class 1
        if 1 in model.classes_:

            phishing_index = list(
                model.classes_
            ).index(1)

            phish_prob = (
                probabilities[
                    phishing_index
                ] * 100
            )

        else:

            # Fallback if model uses boolean/string labels
            phish_prob = (
                probabilities[-1] * 100
            )

        # Keep probability within valid range
        phish_prob = max(
            0.0,
            min(100.0, phish_prob)
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        is_phishing = (
            prediction == 1
            or phish_prob >= 50.0
        )

        # ----------------------------------------------------
        # Probability visual color
        # ----------------------------------------------------

        if phish_prob >= 70:
            probability_color = "#f85149"

        elif phish_prob >= 40:
            probability_color = "#d29922"

        else:
            probability_color = "#3fb950"


        # ====================================================
        # RESULT
        # ====================================================

        st.divider()

        if is_phishing:

            st.markdown(
                """
                <div class="risk-card risk-high">

                    <div class="risk-title">
                        🚨 HIGH RISK: PHISHING URL DETECTED
                    </div>

                    <div class="risk-description">
                        The machine learning model classified this
                        URL as potentially malicious based on its
                        lexical characteristics. Avoid entering
                        passwords, financial information, or other
                        sensitive data.
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                """
                <div class="risk-card risk-low">

                    <div class="risk-title">
                        ✓ LOW RISK: BENIGN URL DETECTED
                    </div>

                    <div class="risk-description">
                        The machine learning model classified this
                        URL as low risk based on its lexical
                        characteristics. Always verify the
                        destination domain before submitting
                        sensitive information.
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


        # ====================================================
        # PROBABILITY
        # ====================================================

        st.markdown(
            """
            <div class="probability-card">

                <div class="probability-label">
                    Phishing Probability
                </div>

                <div class="probability-number">
                    """
            + f"{phish_prob:.2f}%"
            + """
                </div>

                <div class="probability-description">
                    Estimated probability that the URL exhibits
                    characteristics associated with phishing.
                </div>

                <div class="probability-bar">

                    <div
                        class="probability-fill"
                        style="
                            width: """
            + f"{phish_prob:.2f}%"
            + """;
                            background: """
            + probability_color
            + """;
                        "
                    ></div>

                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


        # ====================================================
        # ANALYZED URL
        # ====================================================

        st.markdown(
            '<div class="section-title">Analyzed URL</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="url-card">

                <div class="url-label">
                    Normalized URL
                </div>

                <div class="url-value">
                    {normalized_url}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


        # ====================================================
        # QUICK FEATURES
        # ====================================================

        st.markdown(
            '<div class="section-title">URL Security Indicators</div>',
            unsafe_allow_html=True,
        )

        quick_features = [
            (
                "HTTPS",
                "Enabled"
                if feats["has_https"]
                else "Not detected",
            ),
            (
                "IP Address",
                "Detected"
                if feats["has_ip"]
                else "Not detected",
            ),
            (
                "Suspicious Keywords",
                str(feats["suspicious_words"]),
            ),
            (
                "URL Length",
                str(feats["url_length"]),
            ),
        ]

        cols = st.columns(4)

        for col, (label, value) in zip(
            cols,
            quick_features
        ):

            with col:

                st.markdown(
                    f"""
                    <div class="feature-card">

                        <div class="feature-label">
                            {label}
                        </div>

                        <div class="feature-value">
                            {value}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )


        # ====================================================
        # EXTRACTED FEATURES
        # ====================================================

        st.markdown(
            '<div class="section-title">Extracted URL Features</div>',
            unsafe_allow_html=True,
        )

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
                "Entropy",
            ],

            "Value": [
                str(feats["url_length"]),
                str(feats["num_dots"]),
                (
                    "Yes"
                    if feats["has_https"]
                    else "No"
                ),
                (
                    "Yes"
                    if feats["has_ip"]
                    else "No"
                ),
                str(feats["num_subdirs"]),
                str(feats["num_params"]),
                str(feats["suspicious_words"]),
                str(feats["tld"]),
                str(feats["special_char_count"]),
                str(feats["digits_count"]),
                f"{feats['entropy']:.4f}",
            ],
        }

        df_table = pd.DataFrame(
            feature_table_data
        )

        st.dataframe(
            df_table,
            hide_index=True,
            use_container_width=True,
            height=430,
        )


        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        st.markdown(
            '<div class="section-title">Detection Model</div>',
            unsafe_allow_html=True,
        )

        model_col1, model_col2 = st.columns(2)

        with model_col1:

            st.markdown(
                """
                <div class="model-card">

                    <div class="model-label">
                        MACHINE LEARNING ALGORITHM
                    </div>

                    <div class="model-value">
                        Logistic Regression
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        with model_col2:

            st.markdown(
                """
                <div class="model-card">

                    <div class="model-label">
                        ANALYSIS TYPE
                    </div>

                    <div class="model-value">
                        Lexical URL Analysis
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


        # ====================================================
        # DISCLAIMER
        # ====================================================

        st.markdown(
            """
            <div class="disclaimer">

                <strong>⚠️ Important:</strong>
                Detection results are predictions generated by
                a machine learning model. A high-risk result does
                not guarantee that a website is malicious, and a
                low-risk result does not guarantee complete safety.
                Always verify the destination domain and website
                before entering sensitive information.

            </div>
            """,
            unsafe_allow_html=True,
        )

    except Exception as e:

        st.error(
            "An error occurred while analyzing the URL."
        )

        with st.expander("Technical details"):
            st.code(str(e))


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Phishing URL Detection System •
        Machine Learning-Based URL Security Analysis
    </div>
    """,
    unsafe_allow_html=True,
)
