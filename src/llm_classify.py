import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # works locally

# Fallback to Streamlit secrets if running on Streamlit Cloud
try:
    import streamlit as st
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
except Exception:
    api_key = os.environ.get("GROQ_API_KEY")

client = Groq(api_key=api_key)