import math
import os
import re
from urllib.parse import urlparse

import joblib
import pandas as pd
import streamlit as st
import tldextract


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Phishing URL Detector",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* ==============================
       GLOBAL
       ============================== */

    .stApp {
        background: #0b0f14;
    }

    .main {
        padding-top: 1rem;
    }

    /* Remove default top spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }


    /* ==============================
       HEADER
       ============================== */

    .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 18px 24px;
        background: #11161d;
        border: 1px solid #252c35;
        border-radius: 10px;
        margin-bottom: 25px;
    }

    .brand {
        font-size: 25px;
        font-weight: 700;
        color: #f1f5f9;
        letter-spacing: 0.3px;
    }

    .brand-subtitle {
        color: #7f8a98;
        font-size: 13px;
        margin-top: 3px;
    }

    .system-status {
        color: #58d68d;
        font-size: 13px;
        font-weight: 600;
        padding: 7px 12px;
        border: 1px solid #245b40;
        border-radius: 20px;
        background: #10251b;
    }


    /* ==============================
       SECTION TITLES
       ============================== */

    .section-title {
        color: #f1f5f9;
        font-size: 18px;
        font-weight: 650;
        margin-bottom: 12px;
    }

    .section-line {
        height: 1px;
        background: #252c35;
        margin-bottom: 20px;
    }


    /* ==============================
       INPUT PANEL
       ============================== */

    .panel {
        background: #11161d;
        border: 1px solid #252c35;
        border-radius: 10px;
        padding: 25px;
        min-height: 300px;
    }

    .input-label {
        color: #aab4c0;
        font-size: 13px;
        margin-bottom: 7px;
    }

    .input-description {
        color: #687482;
        font-size: 12px;
        margin-top: 5px;
    }


    /* ==============================
       RISK PANEL
       ============================== */

    .risk-panel {
        background: #11161d;
        border: 1px solid #252c35;
        border-radius: 10px;
        padding: 25px;
        min-height: 300px;
        text-align: center;
    }

    .risk-label {
        color: #7f8a98;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    .risk-score {
        font-size: 58px;
        font-weight: 750;
        margin-top: 15px;
        margin-bottom: 5px;
        color: #ff5c5c;
    }

    .risk-status {
        display: inline-block;
        color: #ff6b6b;
        border: 1px solid #703434;
        background: #281619;
        padding: 7px 18px;
        border-radius: 5px;
        font-size: 13px;
        font-weight: 650;
        letter-spacing: 0.5px;
    }

    .safe-score {
        color: #58d68d;
    }

    .safe-status {
        color: #58d68d;
        border-color: #245b40;
        background: #10251b;
    }


    /* ==============================
       PROGRESS
       ============================== */

    .progress-container {
        margin-top: 25px;
        text-align: left;
    }

    .progress-label {
        color: #8994a1;
        font-size: 12px;
        margin-bottom: 7px;
    }

    .progress-background {
        width: 100%;
        height: 8px;
        background: #252c35;
        border-radius: 10px;
        overflow: hidden;
    }

    .progress-danger {
        height: 100%;
        background: #e05252;
        border-radius: 10px;
    }

    .progress-safe {
        height: 100%;
        background: #43b978;
        border-radius: 10px;
    }


    /* ==============================
       FEATURE CARDS
       ============================== */

    .feature-box {
        background: #11161d;
        border: 1px solid #252c35;
        border-radius: 8px;
        padding: 17px;
        margin-bottom: 10px;
    }

    .feature-name {
        color: #74808d;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }

    .feature-value {
        color: #e7edf4;
        font-size: 19px;
        font-weight: 650;
        margin-top: 5px;
    }


    /* ==============================
       MODEL INFORMATION
       ============================== */

    .model-box {
        background: #11161d;
        border: 1px solid #252c35;
        border-radius: 8px;
        padding: 20px;
    }

    .model-label {
        color: #74808d;
        font-size: 11px;
        text-transform: uppercase;
    }

    .model-value {
        color: #e7edf4;
        font-size: 15px;
        font-weight: 600;
        margin-top: 4px;
    }


    /* ==============================
       URL DISPLAY
       ============================== */

    .url-display {
        background: #080b0f;
        border: 1px solid #252c35;
        padding: 14px 16px;
        border-radius: 7px;
        color: #a9d6ff;
        font-family: monospace;
        font-size: 13px;
        word-break: break-all;
        margin-top: 12px;
        margin-bottom: 20px;
    }


    /* ==============================
       BUTTON
       ============================== */

    .stButton > button {
        width: 100%;
        height: 46px;
        border-radius: 6px;
        border: none;
        background: #2864d7;
        color: white;
        font-weight: 650;
        font-size: 14px;
    }

    .stButton > button:hover {
        background: #3475ec;
        color: white;
    }


    /* ==============================
       FOOTER
       ============================== */

    .footer {
        text-align: center;
        color: #4f5965;
        font-size: 11px;
        margin-top: 35px;
        padding-top: 15px;
        border-top: 1px solid #20262e;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
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

    scaled_path = "artifacts/scaled_columns.joblib"

    if os.path.exists(scaled_path):

        scaled_cols = joblib.load(
            scaled_path
        )

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

    return (
        model,
        scaler,
        feature_cols,
        scaled_cols
    )


try:

    model, scaler, feature_cols, scaled_cols = load_artifacts()

    model_loaded = True

except Exception as e:

    model_loaded = False

    st.error(
        f"Model loading error: {e}"
    )


# ============================================================
# FEATURE FUNCTIONS
# ============================================================

def calculate_entropy(text):

    if not text:
        return 0.0

    probabilities = [
        text.count(c) / len(text)
        for c in set(text)
    ]

    return -sum(
        p * math.log(p, 2)
        for p in probabilities
    )


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

    keywords = [
        "login",
        "verify",
        "update",
        "secure",
        "account",
        "banking",
        "signin",
        "confirm"
    ]

    suspicious_words = sum(
        1
        for word in keywords
        if word in url.lower()
    )

    special_chars = (
        r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,<>\/?]"
    )

    special_char_count = len(
        re.findall(
            special_chars,
            url
        )
    )

    digits_count = sum(
        character.isdigit()
        for character in url
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
        "entropy": entropy
    }


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="header">

    <div>
        <div class="brand">
            PHISHING URL DETECTOR
        </div>

        <div class="brand-subtitle">
            Machine Learning Security Analysis
        </div>
    </div>

    <div class="system-status">
        SYSTEM ONLINE
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# MAIN ANALYZER
# ============================================================

left, right = st.columns(
    [1.25, 1],
    gap="large"
)


# ============================================================
# LEFT PANEL
# ============================================================

with left:

    st.markdown(
        '<div class="section-title">'
        'URL ANALYZER'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-line"></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="input-label">'
        'ENTER WEBSITE URL'
        '</div>',
        unsafe_allow_html=True
    )

    url_input = st.text_input(
        "URL",
        placeholder="https://example.com/login",
        label_visibility="collapsed"
    )

    st.markdown(
        '<div class="input-description">'
        'Enter the complete URL for lexical feature analysis.'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")

    analyze = st.button(
        "ANALYZE URL",
        use_container_width=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# RIGHT PANEL
# ============================================================

with right:

    st.markdown(
        '<div class="section-title">'
        'SECURITY ASSESSMENT'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-line"></div>',
        unsafe_allow_html=True
    )

    if not analyze:

        st.markdown("""
        <div class="risk-panel">

            <div class="risk-label">
                ANALYSIS STATUS
            </div>

            <div style="
                font-size: 28px;
                font-weight: 650;
                margin-top: 35px;
                color: #7f8a98;
            ">
                Awaiting URL
            </div>

            <div style="
                font-size: 13px;
                color: #59636f;
                margin-top: 10px;
            ">
                Enter a URL to begin analysis.
            </div>

        </div>
        """, unsafe_allow_html=True)


# ============================================================
# ANALYSIS
# ============================================================

if analyze:

    if not url_input.strip():

        with right:

            st.error(
                "Please enter a URL."
            )

    elif not model_loaded:

        with right:

            st.error(
                "Machine learning model is unavailable."
            )

    else:

        clean_url = url_input.strip()

        if not re.match(
            r"^https?://",
            clean_url,
            re.IGNORECASE
        ):

            with right:

                st.warning(
                    "Please enter a complete URL beginning "
                    "with http:// or https://."
                )

        else:

            # =================================================
            # EXTRACT FEATURES
            # =================================================

            feats = extract_url_features(
                clean_url
            )

            df_feat = pd.DataFrame(
                [feats]
            )

            df_encoded = pd.get_dummies(
                df_feat
            )

            df_encoded = df_encoded.reindex(
                columns=feature_cols,
                fill_value=0
            )

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

            X_input = df_encoded.to_numpy()

            prediction = model.predict(
                X_input
            )[0]

            probabilities = model.predict_proba(
                X_input
            )[0]

            phish_prob = probabilities[1] * 100

            legitimate_prob = (
                100 - phish_prob
            )

            # =================================================
            # RESULT
            # =================================================

            with right:

                if (
                    prediction == 1
                    or phish_prob > 50
                ):

                    st.markdown(
                        f"""
                        <div class="risk-panel">

                            <div class="risk-label">
                                PHISHING PROBABILITY
                            </div>

                            <div class="risk-score">
                                {phish_prob:.2f}%
                            </div>

                            <div class="risk-status">
                                HIGH RISK
                            </div>

                            <div class="progress-container">

                                <div class="progress-label">
                                    Detection Confidence
                                </div>

                                <div class="progress-background">
                                    <div
                                        class="progress-danger"
                                        style="width:{phish_prob}%"
                                    ></div>
                                </div>

                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        f"""
                        <div class="risk-panel">

                            <div class="risk-label">
                                LEGITIMATE PROBABILITY
                            </div>

                            <div class="risk-score safe-score">
                                {legitimate_prob:.2f}%
                            </div>

                            <div class="risk-status safe-status">
                                LOW RISK
                            </div>

                            <div class="progress-container">

                                <div class="progress-label">
                                    Detection Confidence
                                </div>

                                <div class="progress-background">
                                    <div
                                        class="progress-safe"
                                        style="width:{legitimate_prob}%"
                                    ></div>
                                </div>

                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


            # =================================================
            # ANALYZED URL
            # =================================================

            st.markdown(
                '<div class="section-title" '
                'style="margin-top:35px;">'
                'ANALYZED URL'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="url-display">
                    {clean_url}
                </div>
                """,
                unsafe_allow_html=True
            )


            # =================================================
            # FEATURES
            # =================================================

            st.markdown(
                '<div class="section-title">'
                'URL FEATURE ANALYSIS'
                '</div>',
                unsafe_allow_html=True
            )

            feature_values = [

                ("URL Length", feats["url_length"]),

                ("Number of Dots", feats["num_dots"]),

                (
                    "HTTPS",
                    "YES"
                    if feats["has_https"]
                    else "NO"
                ),

                (
                    "IP Address",
                    "YES"
                    if feats["has_ip"]
                    else "NO"
                ),

                (
                    "Subdirectories",
                    feats["num_subdirs"]
                ),

                (
                    "Parameters",
                    feats["num_params"]
                ),

                (
                    "Suspicious Words",
                    feats["suspicious_words"]
                ),

                (
                    "Top-Level Domain",
                    feats["tld"]
                ),

                (
                    "Special Characters",
                    feats["special_char_count"]
                ),

                (
                    "Digits",
                    feats["digits_count"]
                ),

                (
                    "Entropy",
                    f'{feats["entropy"]:.4f}'
                )
            ]

            # Display four columns
            for row_start in range(
                0,
                len(feature_values),
                4
            ):

                cols = st.columns(4)

                row = feature_values[
                    row_start:row_start + 4
                ]

                for col, item in zip(
                    cols,
                    row
                ):

                    name, value = item

                    with col:

                        st.markdown(
                            f"""
                            <div class="feature-box">

                                <div class="feature-name">
                                    {name}
                                </div>

                                <div class="feature-value">
                                    {value}
                                </div>

                            </div>
                            """,
                            unsafe_allow_html=True
                        )


            # =================================================
            # MODEL INFORMATION
            # =================================================

            st.markdown(
                '<div class="section-title" '
                'style="margin-top:30px;">'
                'MODEL INFORMATION'
                '</div>',
                unsafe_allow_html=True
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.markdown("""
                <div class="model-box">

                    <div class="model-label">
                        Algorithm
                    </div>

                    <div class="model-value">
                        Logistic Regression
                    </div>

                </div>
                """, unsafe_allow_html=True)

            with col2:

                st.markdown("""
                <div class="model-box">

                    <div class="model-label">
                        Analysis
                    </div>

                    <div class="model-value">
                        Lexical Features
                    </div>

                </div>
                """, unsafe_allow_html=True)

            with col3:

                st.markdown("""
                <div class="model-box">

                    <div class="model-label">
                        Output
                    </div>

                    <div class="model-value">
                        Binary Classification
                    </div>

                </div>
                """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

    Phishing URL Detection System |
    Logistic Regression |
    Machine Learning-Based URL Analysis

</div>
""", unsafe_allow_html=True)
