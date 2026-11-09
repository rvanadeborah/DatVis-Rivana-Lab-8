# ============================================================
# LAB 8: QUALITATIVE DATA & SENTIMENT ANALYSIS
# THE VOICE BEHIND THE PRODUCT
# ============================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="The Voice Behind the Product",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background-color: #F7F8F6;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background-color: #EDF2EF;
    border-right: 1px solid #E0E8E2;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] p {
    color: #253D36;
}

h1, h2, h3 {
    color: #253D36 !important;
    font-weight: 600 !important;
}

p, label, li {
    color: #59665F;
}

[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E6EBE7;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 2px 8px rgba(30, 60, 45, 0.03);
}

[data-testid="stMetricLabel"] {
    color: #77847D !important;
}

[data-testid="stMetricValue"] {
    color: #253D36 !important;
    font-weight: 600;
}

div.stButton > button,
div.stDownloadButton > button {
    background-color: #345B4D;
    color: white;
    border: none;
    border-radius: 9px;
    padding: 0.55rem 1.1rem;
    font-weight: 500;
}

div.stButton > button:hover,
div.stDownloadButton > button:hover {
    background-color: #253F35;
    color: white;
    border: none;
}

div[data-baseweb="select"] > div {
    border-radius: 9px;
}

.stTextInput input,
.stTextArea textarea {
    border-radius: 9px;
}

hr {
    border-color: #E2E9E4;
}

[data-testid="stDataFrame"] {
    border: 1px solid #E4EAE5;
    border-radius: 10px;
}

.eyebrow {
    color: #658474;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.section-note {
    background: #EDF3EF;
    border-left: 4px solid #719681;
    padding: 13px 17px;
    border-radius: 5px;
    color: #45594D;
    margin: 12px 0 20px 0;
    font-size: 0.9rem;
}

.small-label {
    color: #789084;
    font-size: 0.75rem;
    letter-spacing: 1.2px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. HEADER IMAGE
# ============================================================

header_path = Path(__file__).parent / "header.png"

if header_path.exists():
    st.image(
        str(header_path),
        use_container_width=True
    )
else:
    st.error(
        "Header image not found. Please place header.png "
        "in the same folder as lab8_sentiment.py."
    )


# ============================================================
# 4. SAMPLE PRODUCT REVIEW DATASET
# ============================================================

reviews = [
    "The bottle keeps my water cold all day.",
    "Great size and easy to carry to class.",
    "I love the simple design and sturdy lid.",
    "It leaks in my bag. Very disappointing.",
    "The cap is hard to open and the bottle feels heavy.",
    "Good value, but cleaning the narrow opening is difficult.",
    "Keeps drinks cold, though the paint scratches easily.",
    "The bottle is fine. Nothing special either way.",
    "I expected better insulation for this price.",
    "Lightweight, useful, and easy to wash.",
    "Not bad, but I would prefer a larger capacity.",
    "Excellent bottle. I use it every day.",
    "The lid broke after one week. Poor quality.",
    "It works as described and arrived on time.",
    "Nice color, but it does not keep drinks cold.",
    "Affordable and practical for carrying water.",
    "The bottle is okay; the handle is uncomfortable.",
    "Very happy with the purchase and the quality."
]


# ============================================================
# 5. PRODUCT TOPIC CLASSIFICATION
# ============================================================

topics = {
    "R01": "Insulation",
    "R02": "Size and Design",
    "R03": "Lid and Design",
    "R04": "Leakage",
    "R05": "Lid and Usability",
    "R06": "Cleaning",
    "R07": "Durability",
    "R08": "Overall Experience",
    "R09": "Insulation and Price",
    "R10": "Cleaning and Usability",
    "R11": "Capacity",
    "R12": "Quality",
    "R13": "Durability and Lid",
    "R14": "Delivery and Performance",
    "R15": "Insulation and Color",
    "R16": "Price",
    "R17": "Comfort",
    "R18": "Quality"
}


df = pd.DataFrame({
    "id": [f"R{i:02}" for i in range(1, 19)],
    "text": reviews
})

df["topic"] = df["id"].map(topics)


# ============================================================
# 6. VADER SENTIMENT ANALYSIS
# ============================================================

analyzer = SentimentIntensityAnalyzer()


def label_sentiment(text):

    score = analyzer.polarity_scores(text)["compound"]

    if score >= 0.05:
        return "Positive", score

    elif score <= -0.05:
        return "Negative", score

    else:
        return "Neutral", score


df[["sentiment", "compound"]] = df["text"].apply(
    lambda text: pd.Series(label_sentiment(text))
)


sentiment_order = [
    "Positive",
    "Neutral",
    "Negative"
]

counts = (
    df["sentiment"]
    .value_counts()
    .reindex(sentiment_order, fill_value=0)
)


palette = {
    "Positive": "#719A83",
    "Neutral": "#D5B878",
    "Negative": "#C9827C"
}


# ============================================================
# 7. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown(
        '<p class="eyebrow">DATA VISUALIZATION</p>',
        unsafe_allow_html=True
    )

    st.markdown("## LAB 08")

    st.caption("Qualitative Data & Sentiment Analysis")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Comment Explorer",
            "Sentiment Lab",
            "Topic Analysis",
            "Findings & Reflection"
        ]
    )

    st.divider()

    st.markdown("**DATASET INFORMATION**")

    st.metric(
        "Total Comments",
        len(df)
    )

    st.caption(
        "Source: Synthetic product review dataset"
    )

    st.divider()

    st.caption("TFB3133 / TEB3133")

    st.caption(
        "Rivana Deborah Satriani | 26010127"
    )


# ============================================================
# 8. PAGE 1: OVERVIEW
# ============================================================

if page == "Overview":

    st.subheader("Dashboard Overview")

    st.markdown(
        '<div class="section-note">'
        'Explore the overall sentiment distribution and discover '
        'the words most frequently used in the product feedback.'
        '</div>',
        unsafe_allow_html=True
    )

    # METRIC CARDS

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Comments",
        len(df)
    )

    col2.metric(
        "Positive",
        int(counts["Positive"])
    )

    col3.metric(
        "Neutral",
        int(counts["Neutral"])
    )

    col4.metric(
        "Negative",
        int(counts["Negative"])
    )

    st.divider()

    # SENTIMENT DISTRIBUTION

    st.markdown("### Sentiment Distribution")

    col1, col2 = st.columns([1.15, 0.85])

    with col1:

        fig, ax = plt.subplots(figsize=(8, 4))

        bars = ax.bar(
            counts.index,
            counts.values,
            color=[
                palette[sentiment]
                for sentiment in counts.index
            ],
            width=0.55
        )

        ax.set_ylabel("Number of Comments")
        ax.set_xlabel("Sentiment Category")
        ax.set_title("Product Review Sentiment")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.grid(
            axis="y",
            alpha=0.2
        )

        for bar in bars:

            height = bar.get_height()

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.1,
                str(int(height)),
                ha="center",
                va="bottom",
                fontsize=10
            )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    with col2:

        st.markdown("#### Sentiment Proportion")

        for sentiment in sentiment_order:

            number = int(counts[sentiment])

            percentage = (
                number / len(df)
            ) * 100

            st.write(f"**{sentiment}**")

            st.progress(
                percentage / 100,
                text=f"{number} comments ({percentage:.1f}%)"
            )

        st.caption(
            "The distribution shows how the sample comments "
            "are classified by the VADER sentiment model."
        )

    st.divider()

    # WORD CLOUD

    st.markdown("### Frequently Used Words")

    st.write(
        "The word cloud highlights frequently occurring terms "
        "across the product reviews."
    )

    all_text = " ".join(
        df["text"].astype(str)
    )

    custom_stopwords = set(STOPWORDS)

    custom_stopwords.update([
        "drinks",
        "drink"
    ])

    wordcloud = WordCloud(
        width=1200,
        height=500,
        background_color="#F7F8F6",
        stopwords=custom_stopwords,
        colormap="Greens",
        max_words=70,
        collocations=False,
        prefer_horizontal=0.9
    ).generate(all_text)

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.imshow(
        wordcloud,
        interpolation="bilinear"
    )

    ax.axis("off")

    plt.tight_layout(pad=0)

    st.pyplot(fig)

    plt.close(fig)

    st.caption(
        "Stop-word adjustment: 'drinks' and 'drink' were removed "
        "because they appeared frequently but provided limited distinction. "
        "'bottle' was retained because it identifies the main product. "
        "Word size represents frequency, not sentiment or importance. "
        "The display is limited to 70 words."
    )

    st.divider()

    # KEY OBSERVATIONS

    st.markdown("### Key Observations")

    most_common = counts.idxmax()

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("**Largest sentiment category**")

        st.markdown(f"### {most_common}")

        st.write(
            f"{int(counts[most_common])} out of {len(df)} comments "
            f"were classified as {most_common.lower()}."
        )

    with col2:

        st.markdown("**Main product feedback areas**")

        st.markdown("### Performance & Usability")

        st.write(
            "The comments discuss insulation, bottle design, "
            "lid usability, cleaning, durability, and product quality."
        )

    st.divider()

    st.markdown(
        '<div class="section-note">'
        'Sentiment classification is an automated interpretation '
        'of text. It should be reviewed alongside the original comments.'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# 9. PAGE 2: COMMENT EXPLORER
# ============================================================

elif page == "Comment Explorer":

    st.subheader("Comment Explorer")

    st.write(
        "Search, filter, and inspect the original product comments "
        "together with their sentiment labels and compound scores."
    )

    st.divider()

    # SEARCH AND FILTERS

    col1, col2, col3 = st.columns([1.3, 1, 1])

    with col1:

        search_term = st.text_input(
            "Search comments",
            placeholder="Type a keyword..."
        )

    with col2:

        sentiment_filter = st.selectbox(
            "Filter by sentiment",
            ["All"] + sentiment_order
        )

    with col3:

        topic_filter = st.selectbox(
            "Filter by topic",
            ["All"] + sorted(df["topic"].unique().tolist())
        )

    filtered_df = df.copy()

    if search_term:

        filtered_df = filtered_df[
            filtered_df["text"].str.contains(
                search_term,
                case=False,
                na=False
            )
        ]

    if sentiment_filter != "All":

        filtered_df = filtered_df[
            filtered_df["sentiment"] == sentiment_filter
        ]

    if topic_filter != "All":

        filtered_df = filtered_df[
            filtered_df["topic"] == topic_filter
        ]

    st.write("")

    st.metric(
        "Matching Comments",
        len(filtered_df)
    )

    display_df = filtered_df[
        [
            "id",
            "text",
            "topic",
            "sentiment",
            "compound"
        ]
    ].copy()

    display_df["compound"] = (
        display_df["compound"].round(3)
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "id": st.column_config.TextColumn(
                "Comment ID"
            ),
            "text": st.column_config.TextColumn(
                "Original Comment",
                width="large"
            ),
            "topic": st.column_config.TextColumn(
                "Main Topic"
            ),
            "sentiment": st.column_config.TextColumn(
                "Sentiment"
            ),
            "compound": st.column_config.NumberColumn(
                "Compound Score",
                format="%.3f"
            )
        }
    )

    st.divider()

    # INDIVIDUAL COMMENT DETAILS

    st.markdown("### Individual Comment Details")

    if not filtered_df.empty:

        selected_id = st.selectbox(
            "Select a comment ID",
            filtered_df["id"].tolist()
        )

        selected_comment = filtered_df[
            filtered_df["id"] == selected_id
        ].iloc[0]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Sentiment",
            selected_comment["sentiment"]
        )

        col2.metric(
            "Compound Score",
            f"{selected_comment['compound']:.3f}"
        )

        col3.metric(
            "Topic",
            selected_comment["topic"]
        )

        st.markdown("**Original customer comment**")

        st.info(
            selected_comment["text"]
        )

        st.caption(
            "Compound score ranges from -1 (most negative) "
            "to +1 (most positive). It is a model output, "
            "not a confidence score."
        )

    else:

        st.warning(
            "No comments match the selected filters."
        )

    st.divider()

    # DOWNLOAD FILTERED COMMENTS

    csv_data = filtered_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Filtered Comments (CSV)",
        data=csv_data,
        file_name="filtered_product_comments.csv",
        mime="text/csv"
    )


# ============================================================
# 10. PAGE 3: SENTIMENT LAB
# ============================================================

elif page == "Sentiment Lab":

    st.subheader("Sentiment Analysis Laboratory")

    st.markdown(
        '<div class="section-note">'
        'Examine the reliability of automated labels by manually '
        'reviewing comments and testing how VADER interprets negation.'
        '</div>',
        unsafe_allow_html=True
    )

    # MANUAL AUDIT

    st.markdown("### Part A — Manual Sentiment Audit")

    st.write(
        "Review two comments from each automated sentiment category. "
        "Compare the VADER classification with your own assessment."
    )

    audit_rows = []

    for sentiment in sentiment_order:

        st.markdown(f"#### {sentiment} Comments")

        candidates = df[
            df["sentiment"] == sentiment
        ].head(2)

        if len(candidates) < 2:

            st.warning(
                f"Only {len(candidates)} comment(s) received a "
                f"{sentiment} label. The dataset does not provide "
                "two examples in this category."
            )

        for _, row in candidates.iterrows():

            with st.container(border=True):

                st.markdown(f"**{row['id']}**")

                st.write(row["text"])

                st.caption(
                    f"VADER: {row['sentiment']} | "
                    f"Compound: {row['compound']:.3f}"
                )

                assessment = st.selectbox(
                    "Your manual assessment",
                    [
                        "Not reviewed",
                        "Seems reasonable",
                        "Borderline",
                        "Incorrect"
                    ],
                    key=f"assessment_{row['id']}"
                )

                explanation = st.text_input(
                    "Reason for your assessment (optional)",
                    key=f"reason_{row['id']}",
                    placeholder="Explain briefly..."
                )

                audit_rows.append({
                    "ID": row["id"],
                    "Original Comment": row["text"],
                    "VADER Label": row["sentiment"],
                    "Compound": row["compound"],
                    "Manual Assessment": assessment,
                    "Explanation": explanation
                })

    audit_df = pd.DataFrame(audit_rows)

    if not audit_df.empty:

        st.divider()

        st.markdown("#### Audit Results")

        st.dataframe(
            audit_df,
            use_container_width=True,
            hide_index=True
        )

        reviewed = (
            audit_df["Manual Assessment"] != "Not reviewed"
        ).sum()

        st.progress(
            reviewed / len(audit_df),
            text=f"{reviewed} of {len(audit_df)} comments reviewed"
        )

    # NEGATION TEST

    st.divider()

    st.markdown("### Part B — Testing Negation")

    st.write(
        "Change the sentence to investigate how negation "
        "affects the sentiment classification."
    )

    test_text = st.text_input(
        "Enter a comment",
        value="The bottle is not easy to clean."
    )

    if test_text.strip():

        test_label, test_score = label_sentiment(
            test_text
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted Sentiment",
                test_label
            )

        with col2:

            st.metric(
                "Compound Score",
                f"{test_score:.3f}"
            )

        st.markdown("**Interpretation**")

        if test_label == "Negative":

            st.info(
                "VADER identifies the sentence as negative. "
                "The phrase 'not easy' changes the meaning "
                "of the original positive expression 'easy to clean'."
            )

        elif test_label == "Positive":

            st.info(
                "VADER classifies this sentence as positive. "
                "Review the wording carefully because negation "
                "may not always be interpreted correctly."
            )

        else:

            st.info(
                "VADER classifies this sentence as neutral. "
                "The result may reflect mixed or weak sentiment."
            )

    else:

        st.warning(
            "Please enter a sentence to test."
        )


# ============================================================
# 11. PAGE 4: TOPIC ANALYSIS
# ============================================================

elif page == "Topic Analysis":

    st.subheader("Sentiment by Product Topic")

    st.markdown(
        '<div class="section-note">'
        'Explore how customer opinions differ across product '
        'features and identify areas that may require improvement.'
        '</div>',
        unsafe_allow_html=True
    )

    # TOPIC SUMMARY TABLE

    st.markdown("### Topic and Sentiment Overview")

    topic_summary = pd.crosstab(
        df["topic"],
        df["sentiment"]
    ).reindex(
        columns=sentiment_order,
        fill_value=0
    )

    st.dataframe(
        topic_summary,
        use_container_width=True
    )

    st.divider()

    # STACKED BAR CHART

    st.markdown("### Sentiment Distribution by Topic")

    fig, ax = plt.subplots(figsize=(12, 6))

    topic_summary.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=[
            palette["Positive"],
            palette["Neutral"],
            palette["Negative"]
        ],
        width=0.7
    )

    ax.set_xlabel("Product Topic")
    ax.set_ylabel("Number of Comments")

    ax.set_title(
        "Sentiment Categories Across Product Topics"
    )

    ax.legend(title="Sentiment")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="x",
        rotation=45
    )

    ax.grid(
        axis="y",
        alpha=0.2
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.caption(
        "Each topic is based on a small number of sample comments. "
        "The chart is useful for exploring patterns, but it should "
        "not be treated as a statistically representative result."
    )

    st.divider()

    # INDIVIDUAL TOPIC EXPLORER

    st.markdown("### Explore an Individual Topic")

    chosen_topic = st.selectbox(
        "Select a product topic",
        sorted(df["topic"].unique().tolist())
    )

    topic_df = df[
        df["topic"] == chosen_topic
    ]

    st.write(
        f"**{len(topic_df)} comment(s)** related to {chosen_topic}."
    )

    st.dataframe(
        topic_df[
            [
                "id",
                "text",
                "sentiment",
                "compound"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    if not topic_df.empty:

        topic_counts = (
            topic_df["sentiment"]
            .value_counts()
            .reindex(
                sentiment_order,
                fill_value=0
            )
        )

        st.markdown("**Sentiment breakdown**")

        for sentiment in sentiment_order:

            value = int(topic_counts[sentiment])

            st.write(
                f"{sentiment}: {value}"
            )

            st.progress(
                value / len(topic_df),
                text=f"{value} of {len(topic_df)} comments"
            )


# ============================================================
# 12. PAGE 5: FINDINGS & REFLECTION
# ============================================================

elif page == "Findings & Reflection":

    st.subheader("Findings, Recommendations & Reflection")

    st.markdown(
        '<div class="section-note">'
        'Communicate the results responsibly by considering '
        'the main findings, limitations, and recommendations.'
        '</div>',
        unsafe_allow_html=True
    )

    # PRODUCT TEAM SUMMARY

    st.markdown("### Part A — Product Team Summary")

    summary = """
The analysis summarizes 18 sample comments about a reusable water bottle. The sentiment chart shows how the feedback is distributed across positive, neutral, and negative categories, while the word cloud highlights frequently used terms. Comments mention product performance and everyday use, including insulation, the lid, cleaning, size, durability, and price. These themes can help a product team identify areas that deserve closer attention.

However, this dataset is small and synthetic, so it does not represent actual customer experiences or the wider market. VADER also uses English-language sentiment patterns and may misread negation, sarcasm, context, or comments that contain both praise and criticism. A single label cannot explain every detail in a mixed comment.

The team should review the original comments alongside the automated labels and investigate recurring concerns such as leakage, insulation, lid function, and cleaning. For a stronger evaluation, the company should collect a larger set of genuine reviews from customers with different usage experiences. It could also ask specific questions about product features and invite optional written explanations to make the feedback more useful.
"""

    word_count = len(summary.split())

    st.markdown(
        '<div class="small-label">PRODUCT TEAM REPORT</div>',
        unsafe_allow_html=True
    )

    with st.container(border=True):

        st.write(summary)

        st.caption(
            f"Word count: {word_count} | Required: 150–200 words"
        )

    if 150 <= word_count <= 200:

        st.success(
            "The summary meets the assignment word requirement."
        )

    else:

        st.warning(
            "Please adjust the summary to 150–200 words."
        )

    # REFLECTION QUESTIONS

    st.divider()

    st.markdown("### Part B — Assignment Reflection")

    reflection_questions = [
        (
            "1. What did the visualizations reveal?",
            "The visualizations reveal common words and the distribution "
            "of positive, neutral, and negative feedback. The word cloud "
            "helps identify frequently mentioned product-related terms, "
            "while the sentiment chart makes category comparisons easier."
        ),
        (
            "2. What could mislead a reader?",
            "Word size does not indicate importance or sentiment, and "
            "automated sentiment labels may overlook mixed opinions, "
            "sarcasm, and negation. A reader could also be misled by "
            "the small synthetic dataset."
        ),
        (
            "3. How would you validate the sentiment labels or collect "
            "more representative feedback?",
            "Sentiment labels should be checked manually against the "
            "original comments. Future analysis should use a larger and "
            "more representative collection of genuine customer feedback."
        )
    ]

    for question, answer in reflection_questions:

        with st.expander(
            question,
            expanded=True
        ):

            st.write(answer)

    # WORD CLOUD IMPROVEMENT

    st.divider()

    st.markdown("### Part C — Word Cloud Improvement")

    st.markdown("""
    **Removed terms:** 'drinks' and 'drink'.

    These words were removed because they are repetitive and
    provide limited information about specific product features.

    **Retained term:** 'bottle'.

    This word was retained because it identifies the product
    being discussed and provides useful context to readers.

    **Visualization limitation:** Larger words represent higher
    frequency, not greater importance, positive sentiment,
    or customer satisfaction.
    """)

    # DOWNLOAD COMPLETE DATASET

    st.divider()

    st.markdown("### Export Final Dataset")

    st.download_button(
        label="Download Complete Lab 8 Dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="lab8_sentiment_complete.csv",
        mime="text/csv"
    )

    st.success(
        "You have reached the end of the Lab 8 dashboard."
    )


# ============================================================
# 13. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align: center;
        padding: 12px 0;
        color: #829087;
        font-size: 0.75rem;
        letter-spacing: 1px;
    ">
        TFB3133 / TEB3133 · DATA VISUALIZATION · LAB 08
        <br>
        QUALITATIVE DATA & SENTIMENT ANALYSIS
    </div>
    """,
    unsafe_allow_html=True
)
