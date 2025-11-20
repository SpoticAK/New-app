# app/main.py
# Ultra-simple Streamlit MVP for your product CSV

import streamlit as st
import pandas as pd
from io import StringIO

st.set_page_config(page_title="Product Discovery — MVP", layout="wide")

# ---------------------------------------------------------
# WEIGHTS FOR SCORING (you can tweak these anytime)
# ---------------------------------------------------------
WEIGHTS = {
    "price": 0.20,
    "sales": 0.25,
    "rating": 0.20,
    "reviews": 0.15,
    "image": 0.20
}

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def normalize_value(val, max_val, invert=False):
    """Returns a 0–100 score."""
    try:
        v = float(val)
    except:
        return 0

    if max_val <= 0:
        return 0

    pct = (v / max_val) * 100
    if invert:
        pct = 100 - pct

    pct = max(0, min(100, round(pct)))
    return pct


def compute_component_scores(row, max_price, max_sales, max_reviews):
    price_score = normalize_value(row["price"], max_price, invert=True)
    sales_score = normalize_value(row["sales_monthly"], max_sales)
    rating_score = max(0, min(100, round((row["rating"] / 5) * 100)))
    reviews_score = normalize_value(row["reviews"], max_reviews)
    image_score = 100  # assume white background for MVP

    return {
        "score_price": int(price_score),
        "score_sales": int(sales_score),
        "score_rating": int(rating_score),
        "score_reviews": int(reviews_score),
        "score_image": int(image_score),
    }


def compute_final_score(c):
    return int(round(
        c["score_price"] * WEIGHTS["price"] +
        c["score_sales"] * WEIGHTS["sales"] +
        c["score_rating"] * WEIGHTS["rating"] +
        c["score_reviews"] * WEIGHTS["reviews"] +
        c["score_image"] * WEIGHTS["image"]
    ))


# ---------------------------------------------------------
# UPLOAD CSV
# ---------------------------------------------------------
st.title("Product Discovery (Ultra-Simple MVP)")
st.write("Upload your CSV with columns: Image, Title, Price, Ratings, Review, Monthly Sales")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if not uploaded:
    st.stop()

try:
    df_raw = pd.read_csv(uploaded)
except Exception as e:
    st.error("Could not read CSV: " + str(e))
    st.stop()

required = {"Image", "Title", "Price", "Ratings", "Review", "Monthly Sales"}
if not required.issubset(set(df_raw.columns)):
    st.error("CSV missing required columns.")
    st.stop()

# ---------------------------------------------------------
# MAP YOUR CSV COLUMNS
# ---------------------------------------------------------
df = df_raw.rename(columns={
    "Image": "image_url",
    "Title": "title",
    "Price": "price",
    "Ratings": "rating",
    "Review": "reviews",
    "Monthly Sales": "sales_monthly",
}).copy()

# Clean numeric fields
for c in ["price", "sales_monthly", "rating", "reviews"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# ---------------------------------------------------------
# GENERATE EXTRA FIELDS
# ---------------------------------------------------------
df.reset_index(inplace=True, drop=False)
df["asin"] = df.apply(lambda r: f"SKU{int(r['index'])+1:05d}", axis=1)
df["category"] = "Fitness"

max_price = df["price"].max() or 1
max_sales = df["sales_monthly"].max() or 1
max_reviews = df["reviews"].max() or 1

rows = []
for _, r in df.iterrows():
    comps = compute_component_scores(r, max_price, max_sales, max_reviews)
    final = compute_final_score(comps)

    row = r.to_dict()
    row.update(comps)
    row["final_score"] = final
    rows.append(row)

df2 = pd.DataFrame(rows)

# ---------------------------------------------------------
# SEARCH + SORT
# ---------------------------------------------------------
q = st.text_input("Search by title")
sort_by = st.selectbox("Sort by", ["final_score", "sales_monthly", "price"])

if q:
    df2 = df2[df2["title"].str.contains(q, case=False, na=False)]

df2 = df2.sort_values(by=sort_by, ascending=False if sort_by!="price" else True)

# ---------------------------------------------------------
# SHOW PRODUCT TABLE
# ---------------------------------------------------------
st.subheader("Products")
st.dataframe(
    df2[["asin", "title", "price", "sales_monthly", "rating", "reviews", "final_score"]],
    height=350
)

# ---------------------------------------------------------
# DETAIL PANEL
# ---------------------------------------------------------
st.subheader("Product Details")
selected = st.selectbox("Select product (ASIN)", [""] + df2["asin"].tolist())

if selected:
    p = df2[df2["asin"] == selected].iloc[0]

    col1, col2 = st.columns([2,1])

    with col1:
        st.image(p["image_url"], width=300)
        st.markdown(f"### {p['title']}")
        st.write(f"Price: ₹{int(p['price'])}")
        st.write(f"Sales/month: {int(p['sales_monthly'])}")
        st.write(f"Rating: {p['rating']}  • Reviews: {int(p['reviews'])}")

        st.markdown("#### Component Scores")
        st.write(f"Price score: {p['score_price']}");   st.progress(p['score_price']/100)
        st.write(f"Sales score: {p['score_sales']}");   st.progress(p['score_sales']/100)
        st.write(f"Rating score: {p['score_rating']}"); st.progress(p['score_rating']/100)
        st.write(f"Reviews score: {p['score_reviews']}"); st.progress(p['score_reviews']/100)
        st.write(f"Image score: {p['score_image']}");   st.progress(p['score_image']/100)

    with col2:
        st.markdown("### Final Score")
        st.metric("Final", p["final_score"])
        st.progress(p["final_score"]/100)

        st.markdown("---")
        st.button("Bookmark")
        st.button("Add Note")

