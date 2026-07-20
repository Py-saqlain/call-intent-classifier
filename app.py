import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from llm_classify import classify_transcript

st.title("Call Intent Classifier")

tab1, tab2 = st.tabs(["Upload File", "Paste Text"])

with tab1:
    uploaded_file = st.file_uploader("Choose a .txt transcript", type="txt")
    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8", errors="ignore")
        st.text_area("Preview", text, height=200)
        if st.button("Classify", key="file_btn"):
            with st.spinner("Classifying..."):
                result = classify_transcript(text)
            st.success(f"Intent: **{result['intent'].upper()}**")
            st.write(f"Confidence: {result['confidence']:.2f}")
            st.write(f"Reason: {result['reason']}")

with tab2:
    pasted_text = st.text_area("Paste transcript here", height=200)
    if st.button("Classify", key="text_btn"):
        if pasted_text.strip():
            with st.spinner("Classifying..."):
                result = classify_transcript(pasted_text)
            st.success(f"Intent: **{result['intent'].upper()}**")
            st.write(f"Confidence: {result['confidence']:.2f}")
            st.write(f"Reason: {result['reason']}")
        else:
            st.warning("Paste some text first")