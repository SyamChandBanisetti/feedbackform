import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
from collections import Counter
import io
import os
import uuid
from dotenv import load_dotenv

# Load env
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load model once
@st.cache_resource
def get_model():
    return genai.GenerativeModel("models/gemini-1.5-flash")

# Analyze sentiment per response using Gemini
def classify_sentiments(responses):
    model = get_model()
    results = []

    for text in responses:
        text = str(text).strip()
        if not text:
            results.append(("NEUTRAL", "Empty response"))
            continue

        prompt = f"""Classify the following response as POSITIVE, NEGATIVE, or NEUTRAL. Also explain why.

Response: "{text[:300]}"

Return format: <SENTIMENT> - <Reason>"""
        try:
            res = model.generate_content(prompt)
            content = res.text.strip()

            if "-" in content:
                label, reason = content.split("-", 1)
                label = label.strip().upper()
                reason = reason.strip()
                if label not in {"POSITIVE", "NEGATIVE", "NEUTRAL"}:
                    label, reason = "NEUTRAL", "Unclear response"
            else:
                label, reason = "NEUTRAL", "Unexpected response"
        except Exception:
            label, reason = "NEUTRAL", "API Error / Limit Reached"

        results.append((label, reason))

    return results

# Generate insights using Gemini, fallback if needed
def summarize_sentiments(column_name, percentages):
    prompt = f"""
You are an educational analyst. Given this sentiment breakdown for the question: "{column_name}" —
Positive: {percentages['Positive']}%, Negative: {percentages['Negative']}%, Neutral: {percentages['Neutral']}%.

Write:
Summary: (1 line)
Insights: (1 line)
Recommendations: (1 line)
"""

    try:
        model = get_model()
        res = model.generate_content(prompt)
        lines = res.text.strip().splitlines()
        summary = insights = recommendations = ""
        for line in lines:
            if "Summary:" in line:
                summary = line.split(":", 1)[-1].strip()
            elif "Insights:" in line:
                insights = line.split(":", 1)[-1].strip()
            elif "Recommendations:" in line:
                recommendations = line.split(":", 1)[-1].strip()
        return summary or "N/A", insights or "N/A", recommendations or "N/A"
    except Exception:
        return ("Summary not available due to API limits",
                "Try reducing response load or upgrade plan.",
                "Re-run analysis later or summarize manually.")

# Page setup
st.set_page_config(page_title="Feedback Form Analyzer", layout="wide")
st.title("📋 Feedback Form Sentiment Analyzer")
uploaded_file = st.file_uploader("📤 Upload CSV Feedback File", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    text_cols = df.select_dtypes(include="object").columns
    exclude_cols = ["email", "name", "timestamp"]
    questions = [col for col in text_cols if col.lower() not in exclude_cols]

    st.success(f"Detected {len(questions)} feedback questions.")

    summary_rows = []

    for idx, col in enumerate(questions):
        st.markdown(f"---\n### Question {idx+1}: {col}")
        responses = df[col].dropna().tolist()
        sentiment_results = classify_sentiments(responses)

        sentiments = [s[0] for s in sentiment_results]
        reasons = [s[1] for s in sentiment_results]
        counter = Counter(sentiments)

        total = len(sentiment_results)
        pos = counter.get("POSITIVE", 0)
        neg = counter.get("NEGATIVE", 0)
        neu = counter.get("NEUTRAL", 0)

        percent = {
            "Positive": round((pos / total) * 100, 1),
            "Negative": round((neg / total) * 100, 1),
            "Neutral": round((neu / total) * 100, 1)
        }

        # Summarize
        summary, insight, reco = summarize_sentiments(col, percent)

        # Chart
        chart_df = pd.DataFrame({
            "Sentiment": ["Positive", "Negative", "Neutral"],
            "Count": [pos, neg, neu]
        })

        unique_key = f"{col}_{uuid.uuid4()}"

        col1, col2 = st.columns([1.5, 2])
        with col1:
            pie = px.pie(chart_df, values="Count", names="Sentiment", title="Sentiment Distribution")
            st.plotly_chart(pie, use_container_width=True, key=f"pie-{unique_key}")
        with col2:
            st.metric("Total Responses", total)
            st.metric("Positive", f"{pos} ({percent['Positive']}%)")
            st.metric("Negative", f"{neg} ({percent['Negative']}%)")
            st.metric("Neutral", f"{neu} ({percent['Neutral']}%)")

        st.markdown(f"**📝 Summary:** {summary}")
        st.markdown(f"**🔍 Insights:** {insight}")
        st.markdown(f"**✅ Recommendations:** {reco}")

        with st.expander("🔍 View Sample Responses"):
            st.dataframe(pd.DataFrame({
                "Response": responses[:10],
                "Sentiment": sentiments[:10],
                "Reason": reasons[:10]
            }), use_container_width=True)

        summary_rows.append({
            "Question": col,
            "Total Responses": total,
            "Positive %": percent["Positive"],
            "Negative %": percent["Negative"],
            "Neutral %": percent["Neutral"],
            "Summary": summary,
            "Insights": insight,
            "Recommendations": reco
        })

    # Downloadable Excel
    if summary_rows:
        st.markdown("### 📥 Download Summary Report")
        summary_df = pd.DataFrame(summary_rows)
        buffer = io.BytesIO()
        summary_df.to_excel(buffer, index=False)
        buffer.seek(0)
        st.download_button("📥 Download Excel", data=buffer,
                           file_name="feedback_summary.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
