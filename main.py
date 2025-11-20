# app/main.py
# Ultra-simple Streamlit MVP (robust parsing) — NO hardcoded CSV paths
# Works with CSV headers: Image, Title, Price, Ratings, Review, Monthly Sales

import streamlit as st
import pandas as pd
import re
from io import StringIO

st.set_page_config(page_title="Product Discovery — MVP", layout="wide")

# -------------------------
# Config / weights
# -------------------------
WEIGHTS = {
    "price": 0.20,
    "sales": 0.25,
    "rating": 0.20,
    "reviews": 0.15,
    "image": 0.20
}

# -------------------------
# Parsing helpers (robust)
# -------------------------
num_re = re.compile(r"[\d,.]+")               # matches numbers with commas/periods
float_re = re.compile(r"(\d+(?:\.\d+)?)")     # matches float like 4.0

def extract_number_int(s):
    if s is None:
        return 0
    s = str(s).strip()
    if s == "" or s.lower() in ["nan", "none"]:
        return 0
    m = num_re.search(s)
    if not m:
        return 0
    raw = m.group(0).replace(",", "")
    try:
        if "." in raw:
            return int(float(raw))
        return int(raw)
    except:
        digits = re.sub(r"[^\d]", "", raw)
        return int(digits) if digits else 0

def extract_number_float(s):
    if s is None:
        return 0.0
    s = str(s).strip()
    if s == "" or s.lower() in ["nan", "none"]:
        return 0.0
    m = float_re.search(s)
    if not m:
        return float(extract_number_int(s))
    try:
        return float(m.group(1))
    except:
        return float(extract_number_int(s))

def parse_price(s):
    if s is None:
        return 0.0
    s = str(s).strip()
    if s == "" or s.lower() in ["nan", "none"]:
        return 0.0
    m = num_re.search(s)
    if not m:
        return 0.0
    raw = m.group(0).replace(",", "")
    try:
        return float(raw)
    except:
        try:
            return float(re.sub(r"[^\d.]", "", raw))
        except:
            return 0.0

# -------------------------
# Scoring helpers
# -------------------------
def normalize_value(val, max_val, invert=False):
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
    rating_score = max(0, min(100, round((row.get("rating", 0) / 5.0) * 100)))
    reviews_score = normalize_value(row["reviews"], max_reviews)
    image_score = 100
    return {
        "score_price": int(price_score),
        "score_sales": int(sales_score),
        "score_rating": int(rating_score),
        "score_reviews": int(reviews_score),
        "score_image": int(image_score),
    }

def compute_final_score(c):
    s = (
        c["score_price"] * WEIGHTS["price"]
        + c["score_sales"] * WEIGHTS["sales"]
        + c["score_rating"] * WEIGHTS["rating"]
        + c["score_reviews"] * WEIGHTS["reviews"]
        + c["score_image"] * WEIGHTS["image"]
    )
    return int(round(s))

# -------------------------
# UI - uploader only (no automatic server file)
# -------------------------
st.title("Product Discovery — Upload CSV (no hidden files)")
st.write("Upload CSV with headers: Image, Title, Price, Ratings, Review, Monthly Sales")

col_left, col_right = st.columns([3,1])

with col_right:
    uploaded = st.file_uploader("Upload products CSV", type=["csv"])
    if st.button("Use tiny example (for quick test)"):
        sample = """Image,Title,Price,Ratings,Review,Monthly Sales
https://via.placeholder.com/240,Adjustable Kettlebell 6kg,699,4.2,420,1450
https://via.placeholder.com/240,Yoga Resistance Band,349,4.0,210,850
"""
        uploaded = StringIO(sample)

with col_left:
    search_q = st.text_input("Search title / ASIN-like id")
    sort_by = st.selectbox("Sort by", ["final_score", "sales_monthly", "price"], index=0)

# require upload
if not uploaded:
    st.info("Please upload your CSV to continue.")
    st.stop()

# -------------------------
# Load CSV (only from uploader)
# -------------------------
try:
    df_raw = pd.read_csv(uploaded, dtype=str)
except Exception as e:
    st.error("Could not read CSV: " + str(e))
    st.stop()

# -------------------------
# Validate columns
# -------------------------
required = {"Image", "Title", "Price", "Ratings", "Review", "Monthly Sales"}
present = set(df_raw.columns)
missing = required - present
if missing:
    st.error("CSV missing required columns: " + ", ".join(sorted(missing)))
    st.stop()

# -------------------------
# Map and clean columns
# -------------------------
df = df_raw.rename(columns={
    "Image": "image_url",
    "Title": "title",
    "Price": "price_raw",
    "Ratings": "rating_raw",
    "Review": "reviews_raw",
    "Monthly Sales": "sales_raw",
}).copy()

df["price"] = df["price_raw"].apply(parse_price)
df["rating"] = df["rating_raw"].apply(extract_number_float)
df["reviews"] = df["reviews_raw"].apply(extract_number_int)
df["sales_monthly"] = df["sales_raw"].apply(extract_number_int)

df.reset_index(inplace=True, drop=False)
df["asin"] = df.apply(lambda r: f"SKU{int(r['index'])+1:06d}", axis=1)
df["category"] = "Fitness"

max_price = float(df["price"].replace(0, pd.NA).dropna().max() or 1)
max_sales = float(df["sales_monthly"].replace(0, pd.NA).dropna().max() or 1)
max_reviews = float(df["reviews"].replace(0, pd.NA).dropna().max() or 1)

computed_rows = []
for _, r in df.iterrows():
    comps = compute_component_scores(r, max_price=max_price, max_sales=max_sales, max_reviews=max_reviews)
    final = compute_final_score(comps)
    row = r.to_dict()
    row.update(comps)
    row["final_score"] = final
    computed_rows.append(row)

df2 = pd.DataFrame(computed_rows)

# -------------------------
# Filtering, searching, sorting
# -------------------------
if search_q:
    df2 = df2[df2["title"].str.contains(search_q, case=False, na=False) | df2["asin"].str.contains(search_q, case=False, na=False)]

if sort_by == "final_score":
    df2 = df2.sort_values(by="final_score", ascending=False)
elif sort_by == "sales_monthly":
    df2 = df2.sort_values(by="sales_monthly", ascending=False)
elif sort_by == "price":
    df2 = df2.sort_values(by="price", ascending=True)

# -------------------------
# Show table & detail
# -------------------------
st.subheader("Products")
st.dataframe(df2[["asin", "title", "price", "sales_monthly", "rating", "reviews", "final_score"]].reset_index(drop=True), height=400)

st.subheader("Product details")
sel = st.selectbox("Select product (ASIN)", options=[""] + df2["asin"].tolist())

if sel:
    p = df2[df2["asin"] == sel].iloc[0]
    c1, c2 = st.columns([2,1])
    with c1:
        st.image(p["image_url"], width=320)
        st.markdown(f"### {p['title']}")
        st.write(f"ASIN: {p['asin']} • Category: {p.get('category','')}")
        st.write(f"Price: ₹{int(p['price'])} • Sales/mo: {int(p['sales_monthly'])} • Rating: {p['rating']} • Reviews: {int(p['reviews'])}")
        st.markdown("**Score breakdown**")
        st.write(f"Price score: {p['score_price']} / 100"); st.progress(p['score_price']/100)
        st.write(f"Sales score: {p['score_sales']} / 100"); st.progress(p['score_sales']/100)
        st.write(f"Rating score: {p['score_rating']} / 100"); st.progress(p['score_rating']/100)
        st.write(f"Reviews score: {p['score_reviews']} / 100"); st.progress(p['score_reviews']/100)
        st.write(f"Image score: {p['score_image']} / 100"); st.progress(p['score_image']/100)
    with c2:
        st.markdown("### Final Score")
        st.metric("Final", p["final_score"])
        st.progress(p["final_score"]/100)
        st.markdown("---")
        st.button("Find suppliers (mock)")
        st.button("Bookmark")
        st.button("Add note")

st.markdown("---")
st.info("This version reads ONLY the CSV you upload in the UI. If you'd like a one-click option to load a server-side file, I can add an explicit button that reads a path you approve — but I won't hardcode any path without your permission.")
