import os
import io
import pandas as pd
import plotly.express as px
import streamlit as st
from collections import Counter
from dotenv import load_dotenv
import google.generativeai as genai
from itertools import zip_longest

# Load Google API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="Gemini Feedback Analyzer", layout="wide")
st.title("🧠 Gemini-Powered Feedback Analyzer")

@st.cache_resource
def get_gemini_model():
    return genai.GenerativeModel("models/gemini-2.0-flash")

def chunked(iterable, size):
    args = [iter(iterable)] * size
    return zip_longest(*args)

def classify_sentiments(texts):
    model = get_gemini_model()
    results = []
    for text in texts:
        text = text.strip()
        if not text:
            results.append(("NEUTRAL", "Empty response"))
            continue
        text = text[:400]
        prompt = f"""You are a sentiment analysis expert. Classify the sentiment of the following text as POSITIVE, NEGATIVE, or NEUTRAL and provide a short reason.

Text: "{text}"

Return format: <label> - <reason>"""
        try:
            response = model.generate_content(prompt).text.strip()
            if "-" in response:
                label, reason = response.split("-", 1)
                label = label.strip().upper()
                reason = reason.strip()
                if label not in {"POSITIVE", "NEGATIVE", "NEUTRAL"}:
                    label, reason = "NEUTRAL", "Unclear response"
            else:
                label, reason = "NEUTRAL", "Invalid format"
        except Exception:
            label, reason = "NEUTRAL", "API Error or limit reached"
        results.append((label, reason))
    return results

def summarize_sentiments(question, percentages):
    model = get_gemini_model()
    prompt = f"""You are a feedback expert. Analyze this sentiment breakdown for the question: "{question}".

Breakdown: {percentages}

Give:
1. Summary
2. Insight
3. Recommendation

Format:
Summary: ...
Insights: ...
Recommendations: ...
"""
    try:
        response = model.generate_content(prompt).text.strip()
        summary = insight = reco = ""
        for line in response.splitlines():
            if line.startswith("Summary:"):
                summary = line.replace("Summary:", "").strip()
            elif line.startswith("Insights:"):
                insight = line.replace("Insights:", "").strip()
            elif line.startswith("Recommendations:"):
                reco = line.replace("Recommendations:", "").strip()
        return summary, insight, reco
    except:
        return "Not available", "API limit reached", "Try again later"

uploaded_file = st.file_uploader("📂 Upload Feedback CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    text_cols = df.select_dtypes(include="object").columns.tolist()
    ignore = ["timestamp", "email", "name", "id"]
    questions = [col for col in text_cols if col.lower() not in ignore]

    st.markdown("### 📊 Question-wise Analysis")
    summary_data = []
    response_data = []

    for group in chunked(questions, 3):
        cols = st.columns(3)
        for i, question in enumerate(group):
            if question is None:
                continue
            with cols[i]:
                st.subheader(f"❓ {question}")
                responses = df[question].dropna().astype(str).tolist()
                sentiments = classify_sentiments(responses)
                labels = [label for label, _ in sentiments]
                reasons = [reason for _, reason in sentiments]

                count = Counter(labels)
                total = len(responses)
                pos = count.get("POSITIVE", 0)
                neg = count.get("NEGATIVE", 0)
                neu = count.get("NEUTRAL", 0)

                percentages = {
                    "Positive": round((pos / total) * 100, 1),
                    "Negative": round((neg / total) * 100, 1),
                    "Neutral": round((neu / total) * 100, 1)
                }

                pie_df = pd.DataFrame({
                    "Sentiment": ["Positive", "Negative", "Neutral"],
                    "Count": [pos, neg, neu]
                })

                pie = px.pie(pie_df, values="Count", names="Sentiment", hole=0.3,
                             title=None, color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(pie, use_container_width=True)

                bar = px.bar(pie_df, x="Sentiment", y="Count", text="Count",
                             color="Sentiment", color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(bar, use_container_width=True)

                summary, insight, reco = summarize_sentiments(question, percentages)

                st.markdown(f"📝 **Summary**: {summary}")
                st.markdown(f"🔎 **Insights**: {insight}")
                st.markdown(f"✅ **Recommendations**: {reco}")

                summary_data.append({
                    "Question": question,
                    "Total": total,
                    "Positive %": percentages["Positive"],
                    "Negative %": percentages["Negative"],
                    "Neutral %": percentages["Neutral"],
                    "Summary": summary,
                    "Insights": insight,
                    "Recommendations": reco
                })

                for r, l, rsn in zip(responses, labels, reasons):
                    response_data.append({
                        "Question": question,
                        "Response": r,
                        "Sentiment": l,
                        "Reason": rsn
                    })

                with st.expander("📋 View Sample Responses"):
                    st.dataframe(pd.DataFrame({
                        "Response": responses,
                        "Sentiment": labels,
                        "Reason": reasons
                    }).head(10), use_container_width=True)

    # Final download section
    st.markdown("## 📥 Download Complete Report")
    summary_df = pd.DataFrame(summary_data)
    response_df = pd.DataFrame(response_data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        response_df.to_excel(writer, sheet_name="Responses", index=False)
    output.seek(0)
    st.download_button("📥 Download Excel Report", data=output,
                       file_name="gemini_feedback_report.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Please upload a CSV file to analyze.")
