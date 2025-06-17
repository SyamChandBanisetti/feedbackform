import os
import io
import pandas as pd
import plotly.express as px
import streamlit as st
from collections import Counter
from dotenv import load_dotenv
from itertools import zip_longest
import openai

# Load OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="OpenAI Feedback Analyzer", layout="wide")
st.title("🧠 OpenAI-Powered Feedback Analyzer")

def chunked(iterable, size):
    args = [iter(iterable)] * size
    return zip_longest(*args)

def classify_sentiments(texts, model="gpt-3.5-turbo"):
    results = []
    for text in texts:
        text = text.strip()
        if not text:
            results.append(("NEUTRAL", "Empty response"))
            continue
        text = text[:400]
        prompt = f"""Classify the sentiment of the following response as POSITIVE, NEGATIVE, or NEUTRAL.
Also explain briefly why.

Response: "{text}"

Respond in the format:
Label: <sentiment>
Reason: <reason>
"""
        try:
            completion = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            content = completion.choices[0].message.content.strip()
            label = "NEUTRAL"
            reason = "Not parsed"

            for line in content.splitlines():
                if line.lower().startswith("label:"):
                    label = line.split(":", 1)[1].strip().upper()
                elif line.lower().startswith("reason:"):
                    reason = line.split(":", 1)[1].strip()

            if label not in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
                label = "NEUTRAL"
                reason = "Invalid classification"
        except Exception as e:
            label = "NEUTRAL"
            reason = f"API Error: {e}"
        results.append((label, reason))
    return results

def summarize_sentiments(question, percentages, model="gpt-3.5-turbo"):
    prompt = f"""Analyze the following feedback sentiment percentages for the question:
"{question}"

Percentages:
Positive: {percentages['Positive']}%
Negative: {percentages['Negative']}%
Neutral: {percentages['Neutral']}%

Provide:
1. Summary
2. Insight
3. Recommendation

Format:
Summary: ...
Insights: ...
Recommendations: ...
"""
    try:
        completion = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        content = completion.choices[0].message.content.strip()
        summary = insight = reco = ""
        for line in content.splitlines():
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
                       file_name="openai_feedback_report.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Please upload a CSV file to analyze.")
