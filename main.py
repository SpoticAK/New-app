# app/main.py
# Product Discovery — Robust Single-file Streamlit MVP
# Works with CSV headers: Image, Title, Price, Ratings, Review, Monthly Sales
#
# Features:
# - Robust parsing (commas in numbers, "4.0 out of 5 stars", "2K+ bought...", currency symbols)
# - K/M handling (K -> *1000, M -> *1_000_000)
# - Missing/blank fields displayed as "NA" in the UI (not 0)
# - Safe normalization & safe progress bars
# - Reads ONLY the CSV uploaded via the Streamlit uploader (no server path)

import math
import re
from io import StringIO

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Product Discovery — Robust MVP", layout="wide")

# -----------------------
# Config
# -----------------------
WEIGHTS = {
    "price": 0.20,
    "sales": 0.25,
    "rating": 0.20,
    "reviews": 0.15,
    "image": 0.20,
}

# -----------------------
# Regex helpers
# -----------------------
NUM_RE = re.compile(r"[\d,.]+")            # matches numbers with commas or dots
FLOAT_RE = re.compile(r"(\d+(?:\.\d+)?)")  # matches float like 4.0

# -----------------------
# Parsing functions
# -----------------------
def extract_number_int(s):
    """
    Extract integer from messy strings.
    Returns an int or None.
    Supports:
      - '1,111' -> 1111
      - '2K+' / '2k' -> 2000
      - '1.5K' -> 1500
      - '3M' -> 3000000
      - '500+' -> 500
      - '' or None -> None
    """
    if s is None:
        return None
    s = str(s).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return None

    lower = s.lower()
    m = NUM_RE.search(s)
    if not m:
        return None
    raw = m.group(0).replace(",", "")  # remove thousand separators

    # If original string contains 'k' or 'm', scale accordingly
    if "k" in lower:
        try:
            return int(float(raw) * 1_000)
        except Exception:
            pass
    if "m" in lower:
        try:
            return int(float(raw) * 1_000_000)
        except Exception:
            pass

    # fallback to integer parse
    try:
        if "." in raw:
            return int(float(raw))
        return int(raw)
    except Exception:
        digits = re.sub(r"[^\d]", "", raw)
        return int(digits) if digits else None


def extract_number_float(s):
    """
    Extract float-like number (e.g., ratings).
    Returns float or None.
    Examples:
      - '4.0 out of 5 stars' -> 4.0
      - '4' -> 4.0
      - '' or None -> None
    """
    if s is None:
        return None
    s = str(s).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return None

    m = FLOAT_RE.search(s)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass

    # fallback to integer extraction
    num = extract_number_int(s)
    return float(num) if num is not None else None


def parse_price(s):
    """
    Parse price-like strings such as '₹699', '1,199', '699.00'.
    Returns float or None.
    """
    if s is None:
        return None
    s = str(s).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return None

    m = NUM_RE.search(s)
    if not m:
        return None
    raw = m.group(0).replace(",", "")
    try:
        return float(raw)
    except Exception:
        cleaned = re.sub(r"[^\d.]", "", raw)
        try:
            return float(cleaned)
        except Exception:
            return None


# -----------------------
# Normalization & scoring
# -----------------------
def normalize(val, max_val, invert=False):
    """
    Normalize a numeric value to 0..100.
    Returns int 0..100 or None if val is missing/invalid.
    Avoids round(nan) errors.
    """
    if val is None:
        return None
    try:
        v = float(val)
    except Exception:
        return None
    if not math.isfinite(v):
        return None

    try:
        max_v = float(max_val)
    except Exception:
        return None
    if not math.isfinite(max_v) or max_v <= 0:
        return None

    pct = (v / max_v) * 100.0
    if invert:
        pct = 100.0 - pct

    if not math.isfinite(pct):
        return None

    return int(max(0, min(100, round(pct))))


def compute_component_scores(row, max_price, max_sales, max_reviews):
    # force numeric conversions (None if invalid)
    try:
        price_val = None if row.get("price") is None else float(row.get("price"))
    except Exception:
        price_val = None

    try:
        sales_raw = row.get("sales_monthly")
        sales_val = None if sales_raw is None else float(sales_raw)
    except Exception:
        sales_val = None

    try:
        rating_raw = row.get("rating")
        rating_val = None if rating_raw is None else float(rating_raw)
    except Exception:
        rating_val = None

    try:
        reviews_raw = row.get("reviews")
        reviews_val = None if reviews_raw is None else float(reviews_raw)
    except Exception:
        reviews_val = None

    price_score = normalize(price_val, max_price, invert=True)
    sales_score = normalize(sales_val, max_sales)
    rating_score = None if rating_val is None else int(max(0, min(100, round((rating_val / 5.0) * 100))))
    reviews_score = normalize(reviews_val, max_reviews)
    image_score = 100

    return {
        "score_price": price_score,
        "score_sales": sales_score,
        "score_rating": rating_score,
        "score_reviews": reviews_score,
        "score_image": image_score,
    }


def compute_final_score(components):
    """
    Weighted sum. If a component is None, treat it as 0 in final score calculation.
    Returns integer 0..100
    """
    total = 0.0
    for key, w in WEIGHTS.items():
        comp_name = f"score_{key}"
        val = components.get(comp_name)
        total += (val if (val is not None) else 0) * w
    return int(round(total))


# -----------------------
# UI helpers
# -----------------------
def display_val(v):
    """Return 'NA' for missing values, else return value (formatted)."""
    if v is None or (isinstance(v, float) and (math.isnan(v) or not math.isfinite(v))):
        return "NA"
    return v


def safe_progress(score):
    """
    Display a progress indicator safely.
    Accepts score in 0..100 or None. If None -> shows 'NA' text.
    """
    if score is None:
        st.write("NA")
        return
    try:
        v = float(score)
    except Exception:
        st.write("NA")
        return
    if not math.isfinite(v):
        st.write("NA")
        return
    v = max(0.0, min(100.0, v))
    st.progress(v / 100.0)


# -----------------------
# Main UI - file upload
# -----------------------
st.title("Product Discovery — Robust MVP")
st.write("Upload CSV with headers: Image, Title, Price, Ratings, Review, Monthly Sales")

col_main, col_actions = st.columns([3, 1])

with col_actions:
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if st.button("Use small example CSV"):
        sample = """Image,Title,Price,Ratings,Review,Monthly Sales
https://via.placeholder.com/240,Adjustable Kettlebell 6kg,699,4.2,420,1450
https://via.placeholder.com/240,Yoga Resistance Band,349,4.0,210,850
"""
        uploaded = StringIO(sample)

with col_main:
    search_q = st.text_input("Search title / ASIN", "")
    sort_by = st.selectbox("Sort by", ["final_score", "sales_monthly", "price"])

if not uploaded:
    st.info("Please upload a CSV file to continue.")
    st.stop()

# -----------------------
# Read CSV (only from uploader)
# -----------------------
try:
    df_raw = pd.read_csv(uploaded, dtype=str)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

required_cols = {"Image", "Title", "Price", "Ratings", "Review", "Monthly Sales"}
missing = required_cols - set(df_raw.columns)
if missing:
    st.error("CSV missing required columns: " + ", ".join(sorted(missing)))
    st.stop()

# -----------------------
# Map & parse columns
# -----------------------
df = df_raw.rename(
    columns={
        "Image": "image_url",
        "Title": "title",
        "Price": "price_raw",
        "Ratings": "rating_raw",
        "Review": "reviews_raw",
        "Monthly Sales": "sales_raw",
    }
).copy()

# parse numeric fields into cleaned columns (None for missing)
df["price"] = df["price_raw"].apply(parse_price)
df["rating"] = df["rating_raw"].apply(extract_number_float)
df["reviews"] = df["reviews_raw"].apply(extract_number_int)
df["sales_monthly"] = df["sales_raw"].apply(extract_number_int)

# generate ASIN-like id and default category
df.reset_index(inplace=True, drop=False)
df["asin"] = df.apply(lambda r: f"SKU{int(r['index'])+1:06d}", axis=1)
df["category"] = df.get("Category", "Fitness")

# -----------------------
# compute normalization maxima (ignore None/NA)
def safe_max(series, default=1):
    s = series.dropna()
    if s.empty:
        return float(default)
    # coerce to numeric safely
    nums = pd.to_numeric(s, errors="coerce").dropna()
    return float(nums.max()) if not nums.empty else float(default)

max_price = safe_max(df["price"], default=1)
max_sales = safe_max(df["sales_monthly"], default=1)
max_reviews = safe_max(df["reviews"], default=1)


# -----------------------
# compute component scores and final score
# -----------------------
computed = []
for _, row in df.iterrows():
    comps = compute_component_scores(row, max_price, max_sales, max_reviews)
    final = compute_final_score(comps)
    rec = row.to_dict()
    rec.update(comps)
    rec["final_score"] = final
    computed.append(rec)

df2 = pd.DataFrame(computed)

# -----------------------
# search & sort
# -----------------------
if search_q:
    df2 = df2[
        df2["title"].str.contains(search_q, case=False, na=False)
        | df2["asin"].str.contains(search_q, case=False, na=False)
    ]

if sort_by == "final_score":
    df2 = df2.sort_values(by="final_score", ascending=False)
elif sort_by == "sales_monthly":
    df2 = df2.sort_values(by="sales_monthly", ascending=False)
elif sort_by == "price":
    df2 = df2.sort_values(by="price", ascending=True)

# -----------------------
# show table with NA display
# -----------------------
def display_cell(val):
    # show 'NA' for missing, else format ints (no .0)
    if val is None or (isinstance(val, float) and not math.isfinite(val)):
        return "NA"
    # if float but whole number, remove .0
    if isinstance(val, float) and val.is_integer():
        return int(val)
    return val



df_display = df2[
    ["asin", "title", "price", "sales_monthly", "rating", "reviews", "final_score"]
].copy()
df_display = df_display.applymap(display_cell)

st.subheader("Products")
st.dataframe(df_display.reset_index(drop=True), height=450)

# -----------------------
# detail panel
# -----------------------
st.subheader("Product details")
sel = st.selectbox("Select product (ASIN)", options=[""] + df2["asin"].tolist())

if sel:
    p = df2[df2["asin"] == sel].iloc[0]

    left, right = st.columns([2, 1])

    with left:
        st.image(p.get("image_url"), width=320)
        st.markdown(f"### {p.get('title')}")
        st.write(f"ASIN: {p.get('asin')}  •  Category: {p.get('category')}")
        st.write(f"Price: {display_cell(p.get('price'))}")
        st.write(f"Sales/month: {display_cell(p.get('sales_monthly'))}")
        st.write(f"Rating: {display_cell(p.get('rating'))}")
        st.write(f"Reviews: {display_cell(p.get('reviews'))}")

        st.markdown("### Component Scores")
        st.write(f"Price score: {display_cell(p.get('score_price'))} / 100")
        safe_progress(p.get("score_price"))

        st.write(f"Sales score: {display_cell(p.get('score_sales'))} / 100")
        safe_progress(p.get("score_sales"))

        st.write(f"Rating score: {display_cell(p.get('score_rating'))} / 100")
        safe_progress(p.get("score_rating"))

        st.write(f"Reviews score: {display_cell(p.get('score_reviews'))} / 100")
        safe_progress(p.get("score_reviews"))

        st.write(f"Image score: {display_cell(p.get('score_image'))} / 100")
        safe_progress(p.get("score_image"))

    with right:
        st.markdown("### Final Score")
        st.metric("Final", p.get("final_score"))
        st.progress(p.get("final_score", 0) / 100.0)

        st.markdown("---")
        st.button("Find suppliers (mock)")
        st.button("Bookmark")
        st.button("Add note")

st.markdown("---")
st.info("Blank fields display as 'NA'. Monthly Sales supports K/M shorthand. If a few rows still look odd, paste 2–3 raw cell examples and I'll tweak the parser.")
