import os
import json
import re

import streamlit as st
from groq import Groq


st.set_page_config(
    page_title="Human Touch Quality Layer",
    page_icon="HT",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_MODEL = "openai/gpt-oss-120b"

AVAILABLE_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

CONTENT_TYPES = [
    "Email",
    "Social-media post",
    "Customer-service response",
    "Business message",
    "Other",
]

# The five dimension weights are controlled by the application.
# This makes the final score reproducible instead of asking the model
# to invent the overall score independently.
SCORE_WEIGHTS = {
    "empathy": 0.20,
    "naturalness": 0.25,
    "personalization": 0.20,
    "context_awareness": 0.25,
    "brand_voice": 0.10,
}

SYSTEM_PROMPT = """
You are an expert communication quality evaluator for a product called
"Human Touch Quality Layer".

Evaluate whether a communication feels genuinely human, thoughtful,
relevant, natural, personalized, context-aware, and appropriate for
its intended audience.

IMPORTANT:
This is NOT a definitive AI detector.
Do not claim that a message was definitely written by AI or definitely
written by a human. Evaluate communication quality only.

Evaluate these five dimensions:

1. EMPATHY
Does the message recognize the recipient's feelings, needs, concerns,
inconvenience, or human impact where appropriate?

2. NATURALNESS
Does it sound authentic and conversational rather than robotic,
generic, repetitive, formulaic, or excessively polished?

3. PERSONALIZATION
Does it contain relevant details that make it feel written for this
recipient or situation? Never reward invented personal information.

4. CONTEXT AWARENESS
Does it appropriately reflect the supplied situation, audience,
purpose, relationship, and requested action?

5. BRAND VOICE
Does it match the supplied brand voice, tone, personality, and level
of professionalism? If no brand voice is supplied, judge general
appropriateness instead.

SCORING:
Each dimension must be an integer from 0 to 100.
Give evidence-based scores. Do not calculate or return an overall score;
the application calculates the overall score from the five dimensions.

Rewrite requirements:
- Preserve the original meaning.
- Preserve factual claims.
- Do not invent facts or personal information.
- Improve human warmth, clarity, naturalness, personalization,
  context awareness, and brand voice where appropriate.

Return ONLY valid JSON in exactly this structure:

{
  "scores": {
    "empathy": 0,
    "naturalness": 0,
    "personalization": 0,
    "context_awareness": 0,
    "brand_voice": 0
  },
  "summary": "Short evidence-based assessment.",
  "potential_concerns": [
    {
      "dimension": "Naturalness",
      "severity": "Low",
      "evidence": "Specific evidence from the message.",
      "explanation": "Why this may reduce human touch."
    }
  ],
  "recommended_changes": [
    "Specific improvement."
  ],
  "recommended_rewrite": "Improved version of the original message."
}

Allowed severity values:
- Low
- Medium
- High
"""


CUSTOM_CSS = """
<style>
.hero {
    padding: 0.8rem 0 1.2rem 0;
}
.hero h1 {
    font-size: 2.55rem;
    margin-bottom: 0.2rem;
}
.hero p {
    color: #666;
    font-size: 1.05rem;
    max-width: 900px;
}
.score-card {
    padding: 1rem;
    border: 1px solid rgba(128,128,128,0.22);
    border-radius: 12px;
    text-align: center;
    margin-bottom: 0.6rem;
}
.score-number {
    font-size: 1.9rem;
    font-weight: 700;
}
.score-label {
    color: #666;
    font-size: 0.9rem;
}
.concern-card {
    padding: 1rem;
    border: 1px solid rgba(128,128,128,0.22);
    border-radius: 10px;
    margin-bottom: 0.8rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def get_api_key():
    """Read the Groq key from Streamlit Secrets or an environment variable."""
    try:
        key = st.secrets.get("GROQ_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY")


def get_client():
    api_key = get_api_key()
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not configured. Add it to Streamlit Secrets "
            "or set it as an environment variable."
        )
    return Groq(api_key=api_key)


def extract_json(text):
    """Extract JSON from plain JSON, fenced JSON, or surrounding text."""
    if not text:
        raise ValueError("The model returned an empty response.")

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(
        r"```(?:json)?\s*(\{.*\})\s*```",
        text,
        re.DOTALL,
    )
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError("Could not extract valid JSON from the model response.")


def clamp_score(value):
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = 0
    return max(0, min(100, value))


def calculate_overall_score(scores):
    """Calculate the reproducible weighted Human-Touch Score."""
    weighted_score = sum(
        clamp_score(scores.get(dimension, 0)) * weight
        for dimension, weight in SCORE_WEIGHTS.items()
    )
    return round(weighted_score)


def verdict_from_score(score):
    if score >= 80:
        return "Strong"
    if score >= 50:
        return "Needs Improvement"
    return "Weak"


def normalize_result(result):
    if not isinstance(result, dict):
        raise ValueError("Model response is not a valid JSON object.")

    raw_scores = result.get("scores", {})
    if not isinstance(raw_scores, dict):
        raw_scores = {}

    scores = {
        "empathy": clamp_score(raw_scores.get("empathy", 0)),
        "naturalness": clamp_score(raw_scores.get("naturalness", 0)),
        "personalization": clamp_score(raw_scores.get("personalization", 0)),
        "context_awareness": clamp_score(
            raw_scores.get("context_awareness", 0)
        ),
        "brand_voice": clamp_score(raw_scores.get("brand_voice", 0)),
    }

    overall = calculate_overall_score(scores)

    concerns = []
    raw_concerns = result.get("potential_concerns", [])
    if isinstance(raw_concerns, list):
        for concern in raw_concerns:
            if not isinstance(concern, dict):
                continue

            severity = str(concern.get("severity", "Low"))
            if severity not in {"Low", "Medium", "High"}:
                severity = "Low"

            concerns.append({
                "dimension": str(concern.get("dimension", "General")),
                "severity": severity,
                "evidence": str(concern.get("evidence", "")),
                "explanation": str(concern.get("explanation", "")),
            })

    changes = result.get("recommended_changes", [])
    if not isinstance(changes, list):
        changes = [str(changes)]

    changes = [str(item) for item in changes if str(item).strip()]

    return {
        "overall_human_touch_score": overall,
        "verdict": verdict_from_score(overall),
        "summary": str(result.get("summary", "")),
        "scores": scores,
        "potential_concerns": concerns,
        "recommended_changes": changes,
        "recommended_rewrite": str(
            result.get("recommended_rewrite", "")
        ),
    }


def evaluate_message(
    message,
    content_type="Other",
    audience="",
    purpose="",
    brand_voice="",
    additional_context="",
    model=DEFAULT_MODEL,
):
    if not message or not message.strip():
        raise ValueError("Message cannot be empty.")

    user_prompt = f"""
Evaluate the following communication.

CONTENT TYPE:
{content_type}

AUDIENCE / RECIPIENT:
{audience or "Not provided"}

PURPOSE:
{purpose or "Not provided"}

BRAND VOICE:
{brand_voice or "Not provided"}

ADDITIONAL CONTEXT:
{additional_context or "Not provided"}

MESSAGE:
{message}

Return only the required JSON object.
"""

    client = get_client()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=3000,
        response_format={"type": "json_object"},
    )

    if not response.choices:
        raise ValueError("The model returned no choices.")

    raw_content = response.choices[0].message.content
    parsed = extract_json(raw_content)
    return normalize_result(parsed)


def display_score_card(label, score):
    st.markdown(
        f"""
        <div class="score-card">
            <div class="score-number">{score}/100</div>
            <div class="score-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_concern(concern):
    st.markdown(
        f"""
        <div class="concern-card">
            <strong>{concern["dimension"]}</strong>
            &nbsp; | &nbsp;
            <strong>{concern["severity"]} Severity</strong>
            <br><br>
            <strong>Evidence:</strong><br>
            {concern["evidence"]}
            <br><br>
            <strong>Why it matters:</strong><br>
            {concern["explanation"]}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>Human Touch Quality Layer</h1>
        <p>
            An AI-powered communication quality-control layer that evaluates
            whether your message feels human, empathetic, natural,
            personalized, context-aware, and aligned with your intended voice.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Evaluation Settings")

    model = st.selectbox(
        "AI Model",
        AVAILABLE_MODELS,
        index=0,
        help="Groq model used for the evaluation.",
    )

    st.divider()

    st.subheader("Communication Context")

    content_type = st.selectbox(
        "Content Type",
        CONTENT_TYPES,
    )

    audience = st.text_input(
        "Audience / Recipient",
        placeholder="Customer, client, colleague...",
    )

    purpose = st.text_input(
        "Purpose",
        placeholder="What should this message achieve?",
    )

    brand_voice = st.text_input(
        "Brand Voice",
        placeholder="Professional, friendly, warm...",
    )

    additional_context = st.text_area(
        "Additional Context",
        placeholder="Relevant background or situation...",
        height=120,
    )

    st.divider()

    st.caption(
        "API credentials are loaded from Streamlit Secrets or "
        "the GROQ_API_KEY environment variable."
    )


# ============================================================
# MESSAGE INPUT
# ============================================================

st.subheader("Communication to Evaluate")

message = st.text_area(
    "Message",
    height=280,
    placeholder=(
        "Paste your email, social-media post, customer-service "
        "response, business message, or other communication here..."
    ),
    label_visibility="collapsed",
)

evaluate_button = st.button(
    "Evaluate Message",
    type="primary",
    use_container_width=True,
)


# ============================================================
# RUN EVALUATION
# ============================================================

if evaluate_button:
    if not message.strip():
        st.warning("Please enter a message before evaluating.")
    else:
        with st.spinner("Analyzing communication quality..."):
            try:
                st.session_state["evaluation_result"] = evaluate_message(
                    message=message,
                    content_type=content_type,
                    audience=audience,
                    purpose=purpose,
                    brand_voice=brand_voice,
                    additional_context=additional_context,
                    model=model,
                )
            except Exception as error:
                st.error(
                    "The evaluation could not be completed. Check your "
                    "API configuration, model availability, network "
                    "connection, and try again."
                )
                with st.expander("Technical details"):
                    st.code(str(error))


# ============================================================
# RESULTS
# ============================================================

if "evaluation_result" in st.session_state:
    result = st.session_state["evaluation_result"]

    st.divider()
    st.subheader("Evaluation Results")

    overall = result["overall_human_touch_score"]

    score_col, verdict_col = st.columns([2, 1])

    with score_col:
        st.metric("Overall Human-Touch Score", f"{overall}/100")
        st.progress(overall / 100)

    with verdict_col:
        st.metric("Verdict", result["verdict"])

    st.info(result["summary"])

    st.subheader("Communication Dimensions")

    scores = result["scores"]

    dimension_data = [
        ("Empathy", scores["empathy"]),
        ("Naturalness", scores["naturalness"]),
        ("Personalization", scores["personalization"]),
        ("Context Awareness", scores["context_awareness"]),
        ("Brand Voice", scores["brand_voice"]),
    ]

    columns = st.columns(5)

    for column, (label, score) in zip(columns, dimension_data):
        with column:
            display_score_card(label, score)
            st.progress(score / 100)

    st.subheader("Potential Concerns")

    concerns = result["potential_concerns"]

    if concerns:
        for concern in concerns:
            display_concern(concern)
    else:
        st.success(
            "No significant communication-quality concerns were identified."
        )

    st.subheader("Recommended Changes")

    if result["recommended_changes"]:
        for change in result["recommended_changes"]:
            st.markdown(f"- {change}")
    else:
        st.write("No major changes recommended.")

    st.subheader("Recommended Rewrite")

    st.text_area(
        "Improved communication",
        value=result["recommended_rewrite"],
        height=300,
        label_visibility="collapsed",
    )

    with st.expander("How the score is calculated"):
        st.write(
            "The AI evaluates five dimensions. The application then "
            "calculates the final score using fixed weights:"
        )

        st.markdown(
            """
            - Empathy: **20%**
            - Naturalness: **25%**
            - Personalization: **20%**
            - Context Awareness: **25%**
            - Brand Voice: **10%**
            """
        )

        st.write(
            f"Current overall score: **{overall}/100**. "
            "The final verdict is Strong at 80–100, "
            "Needs Improvement at 50–79, and Weak below 50."
        )


# ============================================================
# ABOUT
# ============================================================

st.divider()

with st.expander("About Human Touch Quality Layer"):
    st.markdown(
        """
### What is Human Touch Quality Layer?

Human Touch Quality Layer is an AI-powered communication quality-control
tool designed to help people create communication that feels more human,
thoughtful, relevant, and context-aware.

It evaluates:

- Emails
- Social-media posts
- Customer-service responses
- Business messages
- Other professional communication

### Five evaluation dimensions

**Empathy** — recognition of the recipient's needs, concerns, feelings,
inconvenience, or human impact where appropriate.

**Naturalness** — authentic and conversational language rather than
robotic, generic, repetitive, formulaic, or excessively polished wording.

**Personalization** — relevant details that make communication feel
specific to the intended recipient or situation.

**Context Awareness** — alignment with the audience, purpose, relationship,
situation, and requested action.

**Brand Voice** — alignment with the requested tone, personality,
communication style, and professionalism.

### Important limitation

This application is **not a definitive AI detector**.

A high or low Human-Touch Score does not prove whether a message was
written by a human or generated by AI. The score is a communication-quality
signal.

### Responsible use

AI evaluation can miss cultural context, sarcasm, subtle emotional cues,
specialized communication styles, or intentional brevity. Human review is
recommended for legal, medical, financial, employment, or other
high-stakes communication.
"""
    )


st.caption(
    "Human Touch Quality Layer — Communication Quality Control"
)
