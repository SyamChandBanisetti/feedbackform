import streamlit as st
import pandas as pd
from transformers import pipeline
from collections import Counter
import plotly.express as px
import io

@st.cache_resource
def get_sentiment_analyzer():
    return pipeline("sentiment-analysis")

def analyze_sentiment_distribution(series):
    analyzer = get_sentiment_analyzer()
    texts = series.dropna().astype(str).tolist()
    if not texts:
        return None

    results = analyzer(texts)
    sentiment_labels = [res["label"] for res in results]
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

    summary = "⚖️ Mixed Sentiment"
    insights = "Feedback seems varied. Consider qualitative deep-dive."
    recommendations = ""

    if percentages["Positive"] > 70:
        summary = "💚 Highly Positive"
        insights = "Users are very satisfied. Celebrate this and continue!"
    elif percentages["Negative"] > 50:
        summary = "🔴 Mostly Negative"
        insights = "Majority of feedback is negative. Take corrective steps ASAP."
        recommendations = "Investigate top complaints. Conduct surveys/focus groups."
    elif percentages["Neutral"] > 50:
        summary = "🟡 Mostly Neutral"
        insights = "Responses are neutral. Revisit question clarity or engagement."
        recommendations = "Rephrase questions to encourage opinions."

    return {
        "Total": total,
        "Positive": positive,
        "Negative": negative,
        "Neutral": neutral,
        "Percentages": percentages,
        "Summary": summary,
        "Insights": insights,
        "Recommendations": recommendations,
        "Details": list(zip(texts, sentiment_labels))  # For sample display
    }

# App UI
st.set_page_config(page_title="📊 Feedback Sentiment Analyzer", layout="wide")
st.title("📋 CSV Feedback Sentiment Analyzer")

uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    text_columns = df.select_dtypes(include="object").columns.tolist()

    if not text_columns:
        st.warning("No text columns found.")
    else:
        summary_data = []

        for col in text_columns:
            st.markdown(f"---\n### 📌 Column: **{col}**")

            result = analyze_sentiment_distribution(df[col])
            if result:
                col1, col2 = st.columns([1.5, 2])

                # Pie Chart
                with col1:
                    sentiment_data = pd.DataFrame({
                        "Sentiment": ["Positive", "Negative", "Neutral"],
                        "Count": [result["Positive"], result["Negative"], result["Neutral"]]
                    })
                    fig = px.pie(sentiment_data, values="Count", names="Sentiment", title="Sentiment Distribution")
                    st.plotly_chart(fig, use_container_width=True)

                # Metrics + Bar
                with col2:
                    st.metric("🧾 Total Responses", result["Total"])
                    st.metric("✅ Positive", f"{result['Positive']} ({result['Percentages']['Positive']}%)")
                    st.metric("❌ Negative", f"{result['Negative']} ({result['Percentages']['Negative']}%)")
                    st.metric("➖ Neutral", f"{result['Neutral']} ({result['Percentages']['Neutral']}%)")

                    fig2 = px.bar(
                        sentiment_data,
                        x="Sentiment", y="Count", color="Sentiment",
                        title="Sentiment Count", text="Count"
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                st.markdown(f"**Summary**: {result['Summary']}")
                st.markdown(f"**Insights**: {result['Insights']}")
                if result['Recommendations']:
                    st.markdown(f"**Recommendations**: {result['Recommendations']}")

                # Sample Responses
                with st.expander("🔍 View Sample Responses"):
                    sample_df = pd.DataFrame(result["Details"], columns=["Response", "Sentiment"])
                    st.dataframe(sample_df.head(10), use_container_width=True)

                # Collect summary for export
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

        # Download Summary
        if summary_data:
            st.markdown("### 📥 Download Report")
            summary_df = pd.DataFrame(summary_data)
            buffer = io.BytesIO()
            summary_df.to_excel(buffer, index=False)
            buffer.seek(0)

            st.download_button(
                label="📊 Download Excel Report",
                data=buffer,
                file_name="sentiment_analysis_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
