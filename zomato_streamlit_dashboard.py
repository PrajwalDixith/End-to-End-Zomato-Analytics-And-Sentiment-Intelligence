import io
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import streamlit as st

# =====================================================================
# STREAMLIT CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="Zomato EDA & Sentiment Insights",
    page_icon="🍽️",
    layout="wide",
)


# =====================================================================
# DATA LOADING & PREPROCESSING (CACHED)
# =====================================================================
@st.cache_data
def load_and_clean_data():
    reviews_path = r"C:\aFiles And Media\DataAnalytics\HexaSolutions\Hexa Assignments\Assignments To Submit 09-10-2026\Zomato End to End EDA\9- Zomato End to End EDA\Zomato Project\Zomato Restaurant reviews.csv"
    metadata_path = r"C:\aFiles And Media\DataAnalytics\HexaSolutions\Hexa Assignments\Assignments To Submit 09-10-2026\Zomato End to End EDA\9- Zomato End to End EDA\Zomato Project\Zomato Restaurant names and Metadata.csv"

    reviews_df = pd.read_csv(reviews_path)
    metadata_df = pd.read_csv(metadata_path)

    reviews_df.drop_duplicates(inplace=True)
    metadata_df.drop_duplicates(inplace=True)

    # Clean Rating
    reviews_df["Rating_Clean"] = pd.to_numeric(
        reviews_df["Rating"], errors="coerce"
    )
    reviews_df["Rating_Clean"] = reviews_df["Rating_Clean"].fillna(
        reviews_df["Rating_Clean"].median()
    )

    # Clean Cost
    metadata_df["Cost_Clean"] = (
        metadata_df["Cost"].astype(str).str.replace(",", "").str.strip()
    )
    metadata_df["Cost_Clean"] = pd.to_numeric(
        metadata_df["Cost_Clean"], errors="coerce"
    )

    # Text Preprocessing for NLP
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"http\S+|[^a-zA-Z\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    reviews_df["Cleaned_Review"] = reviews_df["Review"].apply(clean_text)

    # Ground truth sentiment proxy
    def to_sentiment(r):
        if r >= 4.0:
            return "Positive"
        elif r >= 3.0:
            return "Neutral"
        return "Negative"

    reviews_df["Sentiment"] = reviews_df["Rating_Clean"].apply(to_sentiment)

    merged_df = pd.merge(
        reviews_df,
        metadata_df,
        left_on="Restaurant",
        right_on="Name",
        how="inner",
    )
    return merged_df


# =====================================================================
# MACHINE LEARNING MODEL TRAINING (CACHED)
# =====================================================================
@st.cache_resource
def train_sentiment_model(df):
    sub = df.dropna(subset=["Cleaned_Review", "Sentiment"])
    tfidf = TfidfVectorizer(
        max_features=2500, ngram_range=(1, 2), stop_words="english"
    )
    X = tfidf.fit_transform(sub["Cleaned_Review"])
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X, sub["Sentiment"])
    return tfidf, model


# =====================================================================
# BULK PDF REPORT GENERATOR (ALL RESTAURANTS)
# =====================================================================
def generate_all_restaurants_pdf(data: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    page_width, page_height = A4

    # -------------------------------------------------------------
    # PAGE 1: OVERALL EXECUTIVE SUMMARY
    # -------------------------------------------------------------
    pdf.setFont("Helvetica-Bold", 20)
    pdf.setFillColor(colors.HexColor("#E23744"))
    pdf.drawString(40, page_height - 50, "Zomato Portfolio Analytics Dossier")

    pdf.setFont("Helvetica", 10)
    pdf.setFillColor(colors.black)
    pdf.drawString(
        40,
        page_height - 70,
        "Comprehensive Performance Summary across all tracked venues",
    )

    total_venues = data["Restaurant"].nunique()
    total_reviews = len(data)
    overall_rating = data["Rating_Clean"].mean()
    pos_ratio = (data["Sentiment"] == "Positive").mean() * 100

    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(
        40,
        page_height - 100,
        f"Total Restaurants: {total_venues}   |   Reviews: {total_reviews:,}   |   Avg Rating: {overall_rating:.2f}   |   Positive: {pos_ratio:.1f}%",
    )

    # Executive Overview Visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2))

    top10 = (
        data.groupby("Restaurant")["Rating_Clean"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    ax1.barh(top10.index[::-1], top10.values[::-1], color="#E23744")
    ax1.set_title("Top 10 Rated Restaurants", fontsize=10, weight="bold")
    ax1.set_xlim(3.5, 5.0)

    sentiment_cts = data["Sentiment"].value_counts(normalize=True).mul(100)
    ax2.pie(
        sentiment_cts.values,
        labels=sentiment_cts.index,
        autopct="%1.1f%%",
        colors=["#2ecc71", "#f39c12", "#e74c3c"],
        startangle=140,
    )
    ax2.set_title("Overall Sentiment Share", fontsize=10, weight="bold")
    plt.tight_layout()

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png", dpi=150)
    plt.close(fig)
    img_buf.seek(0)

    # Fixed: Wrapped with ImageReader
    pdf.drawImage(
        ImageReader(img_buf),
        40,
        page_height - 390,
        width=515,
        height=260,
        preserveAspectRatio=True,
    )

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(
        40, page_height - 415, "Top 5 Establishments by Customer Rating"
    )

    pdf.setFont("Helvetica", 9)
    y_pos = page_height - 435
    top5_summary = (
        data.groupby("Restaurant")
        .agg(
            Rating=("Rating_Clean", "mean"),
            Cost=("Cost_Clean", "first"),
            Reviews=("Review", "count"),
        )
        .sort_values(by="Rating", ascending=False)
        .head(5)
    )

    for idx, (name, row) in enumerate(top5_summary.iterrows(), start=1):
        pdf.drawString(
            50,
            y_pos,
            f"{idx}. {name[:35]} — Avg Rating: {row['Rating']:.2f} | Cost for Two: INR {row['Cost']:.0f} | Reviews: {row['Reviews']}",
        )
        y_pos -= 18

    pdf.showPage()

    # -------------------------------------------------------------
    # PAGES 2+: INDIVIDUAL RESTAURANT CARDS
    # -------------------------------------------------------------
    restaurants = sorted(data["Restaurant"].unique())

    for rest in restaurants:
        sub = data[data["Restaurant"] == rest]
        rest_name = str(rest)[:45]
        avg_rtg = sub["Rating_Clean"].mean()
        cost = sub["Cost_Clean"].iloc[0] if "Cost_Clean" in sub else np.nan
        cuisine = (
            sub["Cuisines"].iloc[0]
            if ("Cuisines" in sub and pd.notna(sub["Cuisines"].iloc[0]))
            else "N/A"
        )
        total_rev = len(sub)
        pos = (sub["Sentiment"] == "Positive").mean() * 100
        neu = (sub["Sentiment"] == "Neutral").mean() * 100
        neg = (sub["Sentiment"] == "Negative").mean() * 100

        pdf.setFont("Helvetica-Bold", 18)
        pdf.setFillColor(colors.HexColor("#2C3E50"))
        pdf.drawString(40, page_height - 50, rest_name)

        pdf.setFont("Helvetica", 10)
        pdf.setFillColor(colors.dimgray)
        pdf.drawString(
            40,
            page_height - 70,
            f"Cuisines: {cuisine[:80]} | Cost for Two: INR {cost:.0f} | Total Reviews: {total_rev}",
        )

        pdf.setFont("Helvetica-Bold", 11)
        pdf.setFillColor(colors.black)
        pdf.drawString(
            40,
            page_height - 95,
            f"Average Rating: {avg_rtg:.2f}/5.0   |   Positive: {pos:.1f}%   |   Neutral: {neu:.1f}%   |   Negative: {neg:.1f}%",
        )

        fig, (p1, p2) = plt.subplots(1, 2, figsize=(7.5, 2.8))

        sub["Rating_Clean"].value_counts().sort_index().plot(
            kind="bar", ax=p1, color="#3498db"
        )
        p1.set_title("Ratings Breakdown", fontsize=9, weight="bold")
        p1.set_xlabel("Stars")
        p1.set_ylabel("Count")

        p2.bar(
            ["Positive", "Neutral", "Negative"],
            [pos, neu, neg],
            color=["#2ecc71", "#f39c12", "#e74c3c"],
        )
        p2.set_title("Sentiment Polarity %", fontsize=9, weight="bold")
        p2.set_ylabel("Share (%)")
        plt.tight_layout()

        img_b = io.BytesIO()
        plt.savefig(img_b, format="png", dpi=150)
        plt.close(fig)
        img_b.seek(0)

        # Fixed: Wrapped with ImageReader
        pdf.drawImage(
            ImageReader(img_b),
            40,
            page_height - 360,
            width=515,
            height=250,
            preserveAspectRatio=True,
        )

        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(
            40, page_height - 380, "Recent Customer Feedback Sample:"
        )

        pdf.setFont("Helvetica-Oblique", 8.5)
        pdf.setFillColor(colors.black)
        cur_y = page_height - 400

        sample_reviews = (
            sub["Review"]
            .dropna()
            .loc[lambda s: s.str.strip().str.len() > 10]
            .head(4)
        )
        for _, text in sample_reviews.items():
            clean_sample = (
                text.replace("\n", " ")[:120] + "..."
                if len(text) > 120
                else text.replace("\n", " ")
            )
            pdf.drawString(45, cur_y, f'• "{clean_sample}"')
            cur_y -= 20

        pdf.showPage()

    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()


# =====================================================================
# DATA INITIALIZATION
# =====================================================================
data = load_and_clean_data()
tfidf_vectorizer, sentiment_model = train_sentiment_model(data)

# =====================================================================
# SIDEBAR CONTROLS & BULK EXPORT
# =====================================================================
st.sidebar.header("Navigation & Filters")
selected_restaurant = st.sidebar.selectbox(
    "Select Restaurant:", options=["All"] + sorted(data["Restaurant"].unique())
)

cost_range = st.sidebar.slider(
    "Filter by Cost for Two (INR):",
    min_value=int(data["Cost_Clean"].min()),
    max_value=int(data["Cost_Clean"].max()),
    value=(int(data["Cost_Clean"].min()), int(data["Cost_Clean"].max())),
)

st.sidebar.divider()
st.sidebar.subheader("Bulk PDF Export")

if "pdf_data" not in st.session_state:
    st.session_state["pdf_data"] = None

if st.sidebar.button("Generate Complete PDF Dossier"):
    with st.spinner("Compiling dossier for all venues... Please wait."):
        st.session_state["pdf_data"] = generate_all_restaurants_pdf(data)
    st.sidebar.success("PDF ready for download!")

if st.session_state["pdf_data"] is not None:
    st.sidebar.download_button(
        label="📥 Download All Restaurants PDF",
        data=st.session_state["pdf_data"],
        file_name="Zomato_All_Restaurants_Analytics_Dossier.pdf",
        mime="application/pdf",
    )

# =====================================================================
# MAIN DASHBOARD INTERFACE
# =====================================================================
filtered_data = data[
    (data["Cost_Clean"] >= cost_range[0])
    & (data["Cost_Clean"] <= cost_range[1])
]
if selected_restaurant != "All":
    filtered_data = filtered_data[
        filtered_data["Restaurant"] == selected_restaurant
    ]

st.title("Zomato Restaurant Analytics & Review Sentiment Dashboard")
st.markdown(
    "An end-to-end analytical view connecting ratings, costs, and customer review sentiment."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Reviews Analyzed", f"{len(filtered_data):,}")
col2.metric("Average Rating", f"{filtered_data['Rating_Clean'].mean():.2f} / 5.0")
col3.metric(
    "Average Cost for Two", f"INR {filtered_data['Cost_Clean'].mean():.0f}"
)
pos_pct = (filtered_data["Sentiment"] == "Positive").mean() * 100
col4.metric("Positive Sentiment Ratio", f"{pos_pct:.1f}%")

st.divider()

# Charts
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.subheader("Customer Rating Distribution")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.histplot(
        filtered_data["Rating_Clean"],
        bins=9,
        kde=True,
        color="red",
        ax=ax,
    )
    ax.set_xlabel("Rating")
    st.pyplot(fig)

with row1_col2:
    st.subheader("Sentiment Breakdown")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sentiment_counts = (
        filtered_data["Sentiment"]
        .value_counts(normalize=True)
        .mul(100)
        .reindex(["Positive", "Neutral", "Negative"])
    )
    sns.barplot(
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        palette=["cyan", "orange", "red"],
        ax=ax,
    )
    ax.set_ylabel("Percentage (%)")
    st.pyplot(fig)

st.divider()

# Real-Time Review Classifier Sandbox
st.subheader("Interactive Review Sentiment Classifier")
st.write(
    "Test raw customer feedback through the trained NLP classification engine:"
)

user_review_input = st.text_area(
    "Enter a sample review text:",
    value="The biryani was flavorful and the service was quick, but the seating area was slightly crowded.",
)

if st.button("Classify Sentiment"):
    cleaned_input = re.sub(
        r"[^a-zA-Z\s]", " ", user_review_input.lower()
    ).strip()
    input_vector = tfidf_vectorizer.transform([cleaned_input])
    prediction = sentiment_model.predict(input_vector)[0]
    probabilities = sentiment_model.predict_proba(input_vector)[0]

    if prediction == "Positive":
        st.success(
            f"Predicted Sentiment: **{prediction}** (Confidence: {max(probabilities):.2%})"
        )
    elif prediction == "Neutral":
        st.warning(
            f"Predicted Sentiment: **{prediction}** (Confidence: {max(probabilities):.2%})"
        )
    else:
        st.error(
            f"Predicted Sentiment: **{prediction}** (Confidence: {max(probabilities):.2%})"
        )