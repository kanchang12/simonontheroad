import json
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from openai import OpenAI

load_dotenv()  # must run before OpenAI() so OPENAI_API_KEY and OPENAI_MODEL come from .env

app = Flask(__name__)
client = OpenAI()
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

SITUATIONS = ["Rough sleeping", "Temporary accommodation", "Hotel", "Housed", "Unknown"]
ETHNICITIES = [
    "Asian or Asian British",
    "Black, Black British, Caribbean or African",
    "Mixed or multiple ethnic groups",
    "White",
    "Other ethnic group",
    "Prefer not to say",
]
FIELD_KEYS = ["name", "alias", "phone", "location", "address", "dob", "age", "gender",
              "ethnicity", "physical", "contact", "ni", "situation"]
CATEGORIES = ["health", "benefits", "need", "followup"]

PROMPT = f"""You read a transcript of a conversation between an outreach worker and a person in need.
Return one JSON object with three keys: "fields", "facts", "summary".

"fields": an object using only these keys: {", ".join(FIELD_KEYS)}.
- Include a key only if it was clearly said in the conversation. Never guess or infer.
- gender and ethnicity: only if the person stated it themselves. Never infer from a name, voice or accent.
- ethnicity must be exactly one of {ETHNICITIES}. situation must be exactly one of {SITUATIONS}.
- dob as YYYY-MM-DD. age as a number.

"facts": a list of objects {{"category": ..., "text": ...}}. category is one of {CATEGORIES}.
- health: conditions, symptoms, injuries, medication. benefits: claims, payments, appointments, problems.
- need: things the person needs (housing, clothing, ID, food). followup: concrete actions agreed by the worker or the person.
- text: a short standalone statement, one fact each.
- You are given a list of already known facts. If a fact means the same as a known one, reuse the known wording exactly.

"summary": at most two plain sentences on what was said. Nothing that was not said.

The transcript may contain speech-recognition errors. Do not correct names by guessing."""


@app.get("/")
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "outreach-prototype.html")


@app.post("/api/extract")
def extract():
    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()
    if not text:
        return jsonify({"fields": {}, "facts": [], "summary": ""})
    known = [str(k) for k in (body.get("known") or [])][:100]
    user = "Known facts:\n" + "\n".join("- " + k for k in known) + "\n\nTranscript:\n" + text
    r = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": user}],
    )
    data = json.loads(r.choices[0].message.content)
    fields = {k: str(v) for k, v in (data.get("fields") or {}).items() if k in FIELD_KEYS and v}
    if fields.get("situation") not in SITUATIONS:
        fields.pop("situation", None)
    if fields.get("ethnicity") not in ETHNICITIES:
        fields.pop("ethnicity", None)
    facts = [
        {"category": f["category"], "text": str(f["text"])}
        for f in (data.get("facts") or [])
        if isinstance(f, dict) and f.get("category") in CATEGORIES and f.get("text")
    ]
    return jsonify({"fields": fields, "facts": facts, "summary": str(data.get("summary") or "")})


if __name__ == "__main__":
    app.run(debug=True)
