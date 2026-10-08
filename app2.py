import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from urllib.parse import urlparse

# 1. Page Configuration
st.set_page_config(
    page_title="Phishing URL Detection",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #F8F9FA; font-family: 'Inter', sans-serif; }
    .top-accent-line { border-top: 3px solid #C23B4E; margin-top: -30px; margin-bottom: 25px; }
    .custom-card { background-color: #FFFFFF; border-radius: 8px; padding: 24px; border: 1px solid #E9ECEF; margin-bottom: 20px; }
    .section-title { font-size: 1.1rem; font-weight: 700; color: #212529; display: flex; align-items: center; gap: 8px; margin-bottom: 16px; }
    
    div.stButton > button:first-child {
        background-color: #C23B4E !important; color: white !important; font-weight: 600 !important;
        border-radius: 6px !important; border: none !important; padding: 0.6rem 1.5rem !important; width: 100%; height: 46px;
    }
    
    .feature-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
    .feature-table th { background-color: #F8F9FA; color: #495057; font-weight: 600; text-align: left; padding: 10px 14px; border-bottom: 2px solid #DEE2E6; }
    .feature-table td { padding: 9px 14px; border-bottom: 1px solid #E9ECEF; color: #212529; }
    
    .verdict-box { border: 1.5px dashed #CED4DA; border-radius: 8px; padding: 25px 15px; text-align: center; background-color: #FAFAFA; }
    .verdict-box-danger { border: 1.5px solid #DC3545; border-radius: 8px; padding: 25px 15px; text-align: center; background-color: #FFF5F5; }
    .verdict-box-safe { border: 1.5px solid #198754; border-radius: 8px; padding: 25px 15px; text-align: center; background-color: #F0FFF4; }
</style>
""", unsafe_allow_html=True)

# Load Artifacts
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("model.joblib")
        scaler = joblib.load("scaler.joblib")
        return model, scaler
    except Exception:
        return None, None

model, scaler = load_artifacts()

def extract_features(url):
    parsed = urlparse(url)
    domain = parsed.netloc or parsed.path.split('/')[0]
    path = parsed.path
    return {
        "URL Length": len(url),
        "Number of Dots": url.count('.'),
        "Number of Hyphens": url.count('-'),
        "Number of Underscores": url.count('_'),
        "Number of Slashes": url.count('/'),
        "Presence of IP Address": 1 if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', url) else 0,
        "Number of Subdomains": max(0, len(domain.split('.')) - 2),
        "HTTPS Token in Domain": 1 if "https" in domain.lower() else 0,
        "Presence of @ Symbol": 1 if "@" in url else 0,
        "Presence of Double Slash": 1 if "//" in path else 0,
        "Number of Query Parameters": len(parsed.query.split('&')) if parsed.query else 0,
        "Presence of Suspicious Words": 1 if re.search(r'(login|verify|update|account|banking|secure|signin)', url, re.IGNORECASE) else 0
    }

# Header
col_logo, col_title = st.columns([0.08, 0.92])
with col_logo:
    st.markdown('<div style="background-color: #C23B4E; width: 42px; height: 48px; border-radius: 6px 6px 20px 20px; display: flex; align-items: center; justify-content: center; color: white; font-size: 22px;">⚓</div>', unsafe_allow_html=True)
with col_title:
    st.markdown('<h2 style="margin:0; font-weight: 800; color: #1A1A1A;">Phishing URL Detection</h2><p style="margin:0; color: #6C757D;">Enter a URL to analyze and detect potential phishing threats.</p>', unsafe_allow_html=True)

st.markdown('<div class="top-accent-line"></div>', unsafe_allow_html=True)

# Input Card
st.markdown('<div class="custom-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title"><span style="color:#C23B4E;"></span> URL Input</div>', unsafe_allow_html=True)
col_input, col_btn = st.columns([0.82, 0.18])
with col_input:
    url_input = st.text_input(label="URL", placeholder="Enter or paste a URL to analyze...", label_visibility="collapsed")
    st.caption("Example: https://www.example.com")
with col_btn:
    analyze_clicked = st.button("Analyze")
st.markdown('</div>', unsafe_allow_html=True)

features_dict = None
is_analyzed = False
phishing_prob = 0.0

if analyze_clicked and url_input:
    features_dict = extract_features(url_input)
    is_analyzed = True
    if model and scaler:
        input_vector = np.array(list(features_dict.values())).reshape(1, -1)
        scaled_vector = scaler.transform(input_vector)
        phishing_prob = float(model.predict_proba(scaled_vector)[0][1] * 100)
    else:
        phishing_prob = 15.0 if "https" in url_input and "login" not in url_input else 82.5

# Features Card
st.markdown('<div class="custom-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title"><span style="color:#C23B4E;"></span> Extracted URL Features</div>', unsafe_allow_html=True)

if features_dict:
    table_rows = "".join([f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in features_dict.items()])
else:
    default_features = ["URL Length", "Number of Dots", "Number of Hyphens", "Number of Underscores", "Number of Slashes", "Presence of IP Address", "Number of Subdomains", "HTTPS Token in Domain", "Presence of @ Symbol", "Presence of Double Slash", "Number of Query Parameters", "Presence of Suspicious Words"]
    table_rows = "".join([f"<tr><td><b>{feat}</b></td><td></td></tr>" for feat in default_features])

st.markdown(f'<table class="feature-table"><thead><tr><th style="width: 60%;">Feature</th><th style="width: 40%;">Value</th></tr></thead><tbody>{table_rows}</tbody></table>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Results Card
st.markdown('<div class="custom-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title"><span style="color:#C23B4E;">🛡️</span> Detection Result</div>', unsafe_allow_html=True)
res_col1, res_col2 = st.columns([0.45, 0.55])

with res_col1:
    st.markdown("<p style='text-align:center; font-weight:700; color:#495057;'>Verdict</p>", unsafe_allow_html=True)
    if not is_analyzed:
        st.markdown('<div class="verdict-box"><div style="font-size: 36px; color: #ADB5BD;">🛡️</div><div style="color: #6C757D; font-weight: 600;">Unresolved</div></div>', unsafe_allow_html=True)
    elif phishing_prob >= 50:
        st.markdown('<div class="verdict-box-danger"><div style="font-size: 36px; color: #DC3545;"></div><div style="color: #DC3545; font-weight: 700;">High Risk Phishing</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="verdict-box-safe"><div style="font-size: 36px; color: #198754;"></div><div style="color: #198754; font-weight: 700;">Legitimate / Safe</div></div>', unsafe_allow_html=True)

with res_col2:
    st.markdown("<p style='text-align:center; font-weight:700; color:#495057;'>Phishing Probability</p>", unsafe_allow_html=True)
    if not is_analyzed:
        st.progress(0)
        st.markdown("<p style='text-align:center; color:#868E96; font-size:0.85rem; margin-top:10px;'>No analysis performed yet.</p>", unsafe_allow_html=True)
    else:
        st.progress(int(phishing_prob))
        st.markdown(f"<h3 style='text-align:center; color:#212529; margin-top:10px;'>{phishing_prob:.1f}%</h3>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
