# app/main.py
# Ultra-simple Streamlit MVP — Clean, Robust Parsing + NA Handling
# Works with CSV headers: Image, Title, Price, Ratings, Review, Monthly Sales

import streamlit as st
import pandas as pd
import re
from io import StringIO

st.set_page_config(page_title="Product Discovery MVP", layout="wide")

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
WEIGHTS = {
    "price": 0.20,
    "sales": 0.25,
    "rating": 0.20,
    "reviews": 0.15,
    "image": 0.20
}

# ---------------------------------------------------
# PARSING HELPERS
# ---------------------------------------------------

num_re = re.compile(r"[\d,.]+")
float_re = re.compile(r"(\d+(?:\.\d+)?)")

def extract_number_int(s):
    """Extract integers from messy strings. Supports '1,111', '2K+', '1.5M', etc."""
    if s is None:
        return None
    s = str(s).strip()
    if s == "" or s.lower() in ["nan", "none"]:
        return None

    lower_s = s.lower()

    # Find first numerical chunk (may include decimal point or comma)
    m = num_re.search(s)
    if not m:
        return None

    raw = m.group(0).replace(",", "")  # e.g. "1,111" -> "1111"

    # Handle K / M formatting
    if "k" in lower_s:
        try:
            return int(float(raw) * 1000)
        except:
            pass
    if "m" in lower_s:
        try:
            return int(float(raw) * 1_000_000)
        except:
            pass

    # Normal number
    try:
        if "." in raw:
            return int(float(raw))
        return int(raw)
    except:
        digits = re.sub(r"[^\d]", "", raw)
        return int(digits) if digits else None


def extract_number_float(s):
    """Extract floats like '4.0 out of 5 stars'."""
    if s is None:
        return None
    s = str(s).strip()
    if s == "" or s.lower() in ["nan", "none"]:
        return None

    m = float_re.search(s)
    if not m:
        num = extract_number_int(s)
        return float(num) if num is not None else None

    try:
        return float(m.group(1))
    except:
        num = extract_number_int(s)
        return float(num) if num is not None else None


def parse_price(s):
    """Extract price from formats like ₹699, 1,199 etc."""
    if s is None:
        return None
    s = str(s).strip()
    if s == "" or s.lower() in ["nan", "none"]:
        return None

    m = num_re.search(s)
    if not m:
        return None

    raw = m.group(0).replace(",", "")
    try:
        return float(raw)
    except:
        try:
            return float(re.sub(r"[^\d.]", "", raw))
        except:
            return None


# ---------------------------------------------------
# SCORING HELPERS
# ---------------------------------------------------

def normalize(val, max_val, invert=False):
    """Normalize to 0–100. If val=None, score=None."""
    if val is None:
        return None
    if max_val <= 0:
        return None

    pct = (float(val) / max_val) * 100
    if invert:
        pct = 100 - pct

    return int(max(0, min(100, round(pct))))


def compute_component_scores(row, max_price, max_sales, max_reviews):
    return {
        "score_price": normalize(row["price"], max_price, invert=True),
        "score_sales": normalize(row["sales_monthly"], max_sales),
        "score_rating": normalize((row["rating"] or 0) * 20, 100),  # rating out of 5 → 0–100
        "score_reviews": normalize(row["reviews"], max_reviews),
        "score_image": 100,  # assume good for MVP
    }


def compute_final_score(c):
    """Final weighted score; if any component None, treat as 0."""
    total = 0
    for key, w in WEIGHTS.items():
        comp_name = f"score_{key}"
        val = c.get(comp_name)
        total += (val if val is not None else 0) * w
    return int(round(total))


# ---------------------------------------------------
# UI — FILE UPLOAD
# ---------------------------------------------------

st.title("Product Discovery — Clean MVP")
st.write("Upload your CSV with headers: **Image, Title, Price, Ratings, Review, Monthly Sales**")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if not uploaded:
    st.info("Please upload your CSV to continue.")
    st.stop()

try:
    df_raw = pd.read_csv(uploaded, dtype=str)
except Exception as e:
    st.error(f"Error reading CSV: {e}")
    st.stop()

required = {"Image", "Title", "Price", "Ratings", "Review", "Monthly Sales"}
missing = required - set(df_raw.columns)
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.stop()

# ---------------------------------------------------
# CLEAN + MAP COLUMNS
# ---------------------------------------------------
df = df_raw.rename(columns={
    "Image": "image_url",
    "Title": "title",
    "Price": "price_raw",
    "Ratings": "rating_raw",
    "Review": "reviews_raw",
    "Monthly Sales": "sales_raw",
}).copy()

# Parse clean numeric fields
df["price"] = df["price_raw"].apply(parse_price)
df["rating"] = df["rating_raw"].apply(extract_number_float)
df["reviews"] = df["reviews_raw"].apply(extract_number_int)
df["sales_monthly"] = df["sales_raw"].apply(extract_number_int)

# Add ASIN-like ID
df.reset_index(inplace=True, drop=False)
df["asin"] = df.apply(lambda r: f"SKU{r['index']+1:06d}", axis=1)
df["category"] = "Fitness"

# ---------------------------------------------------
# NORMALIZATION LIMITS (ignore None)
# ---------------------------------------------------
max_price = df["price"].dropna().max() or 1
max_sales = df["sales_monthly"].dropna().max() or 1
max_reviews = df["reviews"].dropna().max() or 1

# ---------------------------------------------------
# COMPUTE SCORES
# ---------------------------------------------------
rows = []
for _, r in df.iterrows():
    comps = compute_component_scores(r, max_price, max_sales, max_reviews)
    final = compute_final_score(comps)
    row = r.to_dict()
    row.update(comps)
    row["final_score"] = final
    rows.append(row)

df2 = pd.DataFrame(rows)

# ---------------------------------------------------
# SEARCH + SORT
# ---------------------------------------------------
search_q = st.text_input("Search title / ASIN", "")
sort_by = st.selectbox("Sort by", ["final_score", "sales_monthly", "price"])

if search_q:
    df2 = df2[df2["title"].str.contains(search_q, case=False, na=False) |
              df2["asin"].str.contains(search_q, case=False, na=False)]

df2 = df2.sort_values(by=sort_by, ascending=(sort_by == "price"))

# ---------------------------------------------------
# DISPLAY TABLE ("NA" for missing values)
# ---------------------------------------------------
def display(val):
    return "NA" if val is None or pd.isna(val) else val

df_display = df2[["asin", "title", "price", "sales_monthly", "rating", "reviews", "final_score"]].copy()
df_display = df_display.applymap(display)

st.subheader("Products")
st.dataframe(df_display, height=400)

# ---------------------------------------------------
# DETAIL PANEL
# ---------------------------------------------------
st.subheader("Product Details")
selected = st.selectbox("Select product (ASIN)", [""] + df2["asin"].tolist())

if selected:
    p = df2[df2["asin"] == selected].iloc[0]

    col1, col2 = st.columns([2,1])

    with col1:
        st.image(p["image_url"], width=320)
        st.markdown(f"### {p['title']}")
        st.write(f"Price: {display(p['price'])}")
        st.write(f"Sales/month: {display(p['sales_monthly'])}")
        st.write(f"Rating: {display(p['rating'])}")
        st.write(f"Reviews: {display(p['reviews'])}")

        st.markdown("### Component Scores")
        for key in ["price", "sales", "rating", "reviews", "image"]:
            score = p.get(f"score_{key}")
            st.write(f"{key.capitalize()} score: {display(score)}")
            if score is not None:
                st.progress(score / 100)

    with col2:
        st.markdown("### Final Score")
        st.metric("Final", p["final_score"])
        st.progress(p["final_score"] / 100)
        st.markdown("---")
        st.button("Bookmark")
        st.button("Add Note")

st.markdown("---")
st.info("Blank fields now show 'NA'. Monthly Sales handles K/M formats. Ratings, Reviews, Price are fully cleaned.")
