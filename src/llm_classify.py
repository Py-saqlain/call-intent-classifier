import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # reads .env file and loads into os.environ

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

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
    import json
    return json.loads(response.choices[0].message.content)


if __name__ == "__main__":
    with open("sample_transcript.txt", "r", encoding="utf-8") as f:
        text = f.read()
    result = classify_transcript(text)
    print(result)