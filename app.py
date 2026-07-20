import streamlit as st
import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Works both locally (.env) and on Streamlit Cloud (Secrets)
try:
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
except Exception:
    api_key = os.environ.get("GROQ_API_KEY")

client = Groq(api_key=api_key)

SYSTEM_PROMPT = """You are a call intent classifier. Read the call transcript and classify it into EXACTLY ONE of these 4 categories:

- booking: caller wants to make a new reservation, order, appointment, or ride
- transfer: caller wants to be connected to a human agent, another department, or a specific person
- cancellation: caller wants to cancel an existing booking, order, subscription, or appointment
- modification: caller wants to change, reschedule, or update an existing booking, order, or appointment

Rules:
- Base your answer on what the CALLER actually wants, not just isolated keywords.
- If the call ends with the AGENT transferring the caller due to a technical issue or error (not because the caller asked to be transferred), classify based on the caller's original request, not the ending.
- Respond with ONLY valid JSON, no extra text: {"intent": "booking|transfer|cancellation|modification", "confidence": 0.0-1.0, "reason": "one short sentence"}
"""

def classify_transcript(text: str) -> dict:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)


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