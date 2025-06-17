import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
from collections import Counter
import io
import os
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

@st.cache_resource
def get_gemini_model():
    return genai.GenerativeModel("models/gemini-2.0-flash")

def analyze_sentiments_with_gemini(texts):
    model = get_gemini_model()
    all_results = []

    for text in texts:
        text = text.strip()
        if not text:
            all_results.append(("NEUTRAL", "Empty response"))
            continue

        text = text[:400]
        prompt = f"""
You are a sentiment analysis expert. Classify the sentiment of the following text as POSITIVE, NEGATIVE, or NEUTRAL and provide a brief reason.

Text: "{text}"

Return format: <label> - <reason>
"""
        try:
            response = model.generate_content(prompt)
            content = response.text.strip()

            if "-" in content:
                label, reason = content.split("-", 1)
                label = label.strip().upper()
                reason = reason.strip()
                if label not in {"POSITIVE", "NEGATIVE", "NEUTRAL"}:
                    label, reason = "NEUTRAL", "Unclear response"
            else:
                label, reason = "NEUTRAL", "Invalid format"
        except Exception:
            label, reason = "NEUTRAL", "API Error"

        all_results.append((label, reason))
    return all_results

def analyze_sentiment_distribution(series, max_rows=None):
    texts = series.dropna().astype(str).tolist()
    if max_rows:
        texts = texts[:max_rows]

    if not texts:
        return None

    results = analyze_sentiments_with_gemini(texts)
    sentiment_labels = [res[0] for res in results]
    reasons = [res[1] for res in results]

    counts = Counter(sentiment_labels)
    total = len(results)
    positive = counts.get("POSITIVE", 0)
    negative = counts.get("NEGATIVE", 0)
    neutral = counts.get("NEUTRAL", 0)

    percentages = {
        "Positive": round((positive / total) * 100, 1),
        "Negative": round((negative / total) * 100, 1),
        "Neutral": round((neutral / total) * 100, 1)
    }

    summary_prompt = f"""
Analyze this sentiment breakdown: {percentages}.
Give a short:
1. Summary
2. Insight
3. Practical recommendation

Return format:
Summary: ...
Insights: ...
Recommendations: ...
"""
    try:
        response = get_gemini_model().generate_content(summary_prompt)
        response_text = response.text.strip()
    except:
        response_text = "Summary: Not available\nInsights: API limit reached\nRecommendations: Try again later"

    summary = insights = recommendations = ""
    for line in response_text.splitlines():
        if line.startswith("Summary:"):
            summary = line.replace("Summary:", "").strip()
        elif line.startswith("Insights:"):
            insights = line.replace("Insights:", "").strip()
        elif line.startswith("Recommendations:"):
            recommendations = line.replace("Recommendations:", "").strip()

    return {
        "Total": total,
        "Positive": positive,
        "Negative": negative,
        "Neutral": neutral,
        "Percentages": percentages,
        "Summary": summary,
        "Insights": insights,
        "Recommendations": recommendations,
        "Details": list(zip(texts, sentiment_labels, reasons))
    }

# 🌟 Streamlit UI
st.set_page_config(page_title="📊 Gemini Feedback Analyzer", layout="wide")
st.title("🧠 Gemini-Powered CSV Feedback Analyzer")

uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    text_columns = df.select_dtypes(include="object").columns.tolist()

    ignore_cols = ["timestamp", "email", "id", "name"]
    text_columns = [col for col in text_columns if col.lower() not in ignore_cols]

    analyze_all = st.checkbox("📈 Analyze all rows?", value=True)
    max_rows = None if analyze_all else st.slider("🔢 Max Responses to Analyze per Question", 10, 100, 30)

    if not text_columns:
        st.warning("No suitable text columns found.")
    else:
        summary_data = []

        for col in text_columns:
            st.markdown(f"---\n### 📌 Column: **{col}**")
            result = analyze_sentiment_distribution(df[col], max_rows=max_rows)

            if result:
                col1, col2 = st.columns([1.5, 2])

                sentiment_data = pd.DataFrame({
                    "Sentiment": ["Positive", "Negative", "Neutral"],
                    "Count": [result["Positive"], result["Negative"], result["Neutral"]]
                })

                with col1:
                    pie = px.pie(sentiment_data, values="Count", names="Sentiment", title="Sentiment Distribution")
                    st.plotly_chart(pie, use_container_width=True, key=f"pie-{col}")

                with col2:
                    st.metric("🧾 Total", result["Total"])
                    st.metric("✅ Positive", f"{result['Positive']} ({result['Percentages']['Positive']}%)")
                    st.metric("❌ Negative", f"{result['Negative']} ({result['Percentages']['Negative']}%)")
                    st.metric("➖ Neutral", f"{result['Neutral']} ({result['Percentages']['Neutral']}%)")

                    bar = px.bar(sentiment_data, x="Sentiment", y="Count", color="Sentiment", text="Count")
                    st.plotly_chart(bar, use_container_width=True, key=f"bar-{col}")

                st.markdown(f"**📝 Summary**: {result['Summary']}")
                st.markdown(f"**🔎 Insights**: {result['Insights']}")
                st.markdown(f"**✅ Recommendations**: {result['Recommendations']}")

                with st.expander("🔍 View Sample Responses & Reasoning"):
                    sample_df = pd.DataFrame(result["Details"], columns=["Response", "Sentiment", "Reason"])
                    st.dataframe(sample_df.head(10), use_container_width=True)

                summary_data.append({
                    "Column": col,
                    "Total Responses": result["Total"],
                    "Positive %": result['Percentages']['Positive'],
                    "Negative %": result['Percentages']['Negative'],
                    "Neutral %": result['Percentages']['Neutral'],
                    "Summary": result["Summary"],
                    "Insights": result["Insights"],
                    "Recommendations": result["Recommendations"]
                })

        if summary_data:
            st.markdown("### 📥 Download Report")
            summary_df = pd.DataFrame(summary_data)
            buffer = io.BytesIO()
            summary_df.to_excel(buffer, index=False)
            buffer.seek(0)

            st.download_button(
                label="📊 Download Excel Report",
                data=buffer,
                file_name="gemini_sentiment_analysis_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
