# app/main.py
# Product Discovery — MVP with YOUR custom scoring system

import math
import re
from io import StringIO

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Product Discovery — MVP", layout="wide")

# -----------------------
# CONFIG: WEIGHTS / MAX POINTS
# -----------------------
# These are BOTH the max score per component and the "weights"
WEIGHTS = {
    "price": 20,
    "reviews": 15,
    "rating": 20,
    "sales": 25,
}

TOTAL_MAX_POINTS = sum(WEIGHTS.values())  # 80

# -----------------------
# REGEX HELPERS
# -----------------------
NUM_RE = re.compile(r"[\d,.]+")            # numbers with commas / dots
FLOAT_RE = re.compile(r"(\d+(?:\.\d+)?)")  # float like 4.0


# -----------------------
# PARSING FUNCTIONS
# -----------------------
def extract_number_int(s):
    """
    Extract integer from messy strings.
    Supports:
      '1,111' -> 1111
      '2K+' / '2k' -> 2000
      '1.5K' -> 1500
      '500+ bought in past month' -> 500
    Does NOT support millions (M ignored completely).
    """
    if s is None:
        return None
    s = str(s).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return None

    # find the first numeric chunk
    m = NUM_RE.search(s)
    if not m:
        return None

    raw = m.group(0).replace(",", "")  # "1,234" -> "1234"
    end_pos = m.end()

    # look at the immediate suffix right after the number
    suffix = s[end_pos:].lstrip()

    # ONLY accept k / K as thousand, ignore M completely
    multiplier = 1
    if suffix:
        first = suffix[0].lower()
        if first == "k":
            multiplier = 1000

    try:
        base = float(raw)
    except Exception:
        digits = re.sub(r"[^\d.]", "", raw)
        if digits == "":
            return None
        base = float(digits)

    return int(base * multiplier)


def extract_number_float(s):
    """
    Extract float-like number from messy text.
    e.g. '4.0 out of 5 stars' -> 4.0
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

    n = extract_number_int(s)
    return float(n) if n is not None else None


def parse_price(s):
    """
    Parse price like '₹249', '1,199', '299.00'
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


def parse_rating(s):
    """
    Parse rating like '4.0 out of 5 stars' -> 4.0 (clamped to 0–5)
    """
    v = extract_number_float(s)
    if v is None:
        return None
    return max(0.0, min(5.0, v))


def parse_sales(s):
    """Parse monthly sales strings like '3K+ bought in past month'."""
    return extract_number_int(s)


def parse_reviews(s):
    """Parse review count strings like '1,111' or '2K+'."""
    return extract_number_int(s)


# -----------------------
# SCORING: YOUR BRACKET RULES
# -----------------------

# PRICE: (max 20)
# <200 = 5
# 200 - 250 = 10
# 250 - 300 = 15
# >300 = 20
def score_price(price):
    if price is None:
        return None
    try:
        p = float(price)
    except Exception:
        return None

    if p < 200:
        return 5
    elif p <= 250:
        return 10
    elif p <= 300:
        return 15
    else:
        return 20


# REVIEW: (max 15)
# <50 = 5
# 50-500 = 10
# 500 - 1000 = 15
# >1000 = 5
def score_reviews(reviews):
    if reviews is None:
        return None
    try:
        r = int(reviews)
    except Exception:
        return None

    if r < 50:
        return 5
    elif r <= 500:
        return 10
    elif r <= 1000:
        return 15
    else:
        return 5


# RATING (max 20)
# <3.4 = 5
# 3.4-3.7 = 10
# 3.7-4.0 = 15
# >4.0 = 20
def score_rating(rating):
    if rating is None:
        return None
    try:
        r = float(rating)
    except Exception:
        return None

    if r < 3.4:
        return 5
    elif r <= 3.7:
        return 10
    elif r <= 4.0:
        return 15
    else:
        return 20


# MONTHLY SALES (max 25)
# <100 = 5
# 100-200 = 15
# 200 - 500 = 20
# 500-1000 = 25
# 1000 - 5000 = 10
# >5000 = 5
def score_sales(sales):
    if sales is None:
        return None
    try:
        s = int(sales)
    except Exception:
        return None

    if s < 100:
        return 5
    elif s <= 200:
        return 15
    elif s <= 500:
        return 20
    elif s <= 1000:
        return 25
    elif s <= 5000:
        return 10
    else:
        return 5


# -----------------------
# FINAL SCORE CALCULATION
# -----------------------
def compute_scores_row(row):
    """
    Compute component scores + final_score for a single row dict.
    Handles missing fields by:
      - NA for that component
      - excluding that component's max points from available_weight
      - scaling final_score back to TOTAL_MAX_POINTS (80)
    """
    scores = {}
    available_weight = 0
    obtained_points = 0

    # PRICE
    sp = score_price(row.get("price"))
    scores["score_price"] = sp
    if sp is not None:
        available_weight += WEIGHTS["price"]
        obtained_points += sp

    # REVIEWS
    srv = score_reviews(row.get("reviews"))
    scores["score_reviews"] = srv
    if srv is not None:
        available_weight += WEIGHTS["reviews"]
        obtained_points += srv

    # RATING
    srt = score_rating(row.get("rating"))
    scores["score_rating"] = srt
    if srt is not None:
        available_weight += WEIGHTS["rating"]
        obtained_points += srt

    # SALES
    ss = score_sales(row.get("sales_monthly"))
    scores["score_sales"] = ss
    if ss is not None:
        available_weight += WEIGHTS["sales"]
        obtained_points += ss

    # Final score scaling
    if available_weight == 0:
        final_score = None
    else:
        final_score = int(round(obtained_points / available_weight * TOTAL_MAX_POINTS))

    scores["final_score"] = final_score
    return scores


# -----------------------
# UI HELPERS
# -----------------------
def display_cell(val):
    """Show '-' for missing, remove .0 from whole floats."""
    # treat real None / NaN / 'none' / 'nan' / '' all as missing
    if (
        val is None
        or (isinstance(val, float) and not math.isfinite(val))
        or (isinstance(val, str) and val.strip().lower() in ("none", "nan", "na", ""))
    ):
        return "-"
    if isinstance(val, float) and val.is_integer():
        return int(val)
    return val



def safe_progress(score, max_points):
    """Safe progress bar for a component with given max_points."""
    if score is None:
        st.write("-")
        return
    try:
        v = float(score)
    except Exception:
        st.write("NA")
        return
    if not math.isfinite(v):
        st.write("-")
        return
    v = max(0.0, min(max_points, v))
    st.progress(v / max_points)


def format_value(val):
    """Formatter for table display: '-', remove .0, keep sorting numeric."""
    if (
        val is None
        or (isinstance(val, float) and not math.isfinite(val))
        or (isinstance(val, str) and val.strip().lower() in ("none", "nan", "na", ""))
    ):
        return "-"
    if isinstance(val, float) and val.is_integer():
        return str(int(val))
    return str(val)



# -----------------------
# MAIN APP
# -----------------------
def main():
    st.title("Product Discovery — MVP (Custom Scoring)")
    st.write("Upload CSV with columns: **Image, Title, Price, Ratings, Review, Monthly Sales**")

    col_main, col_side = st.columns([3, 1])

    with col_side:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if st.button("Use tiny example CSV"):
            sample = """Image,Title,Price,Ratings,Review,Monthly Sales
https://via.placeholder.com/240,Adjustable Kettlebell 6kg,₹249,4.2 out of 5 stars,1,234,2K+ bought in past month
https://via.placeholder.com/240,Yoga Resistance Band,₹320,3.8 out of 5 stars,340,500+ bought in past month
"""
            uploaded = StringIO(sample)

    with col_main:
        search_q = st.text_input("Search Title / ASIN", "")
        sort_by = st.selectbox("Sort by", ["final_score", "sales_monthly", "price"])

    if not uploaded:
        st.info("Upload a CSV to begin.")
        return

    # Read CSV
    try:
        df_raw = pd.read_csv(uploaded, dtype=str)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return

    required = {"Image", "Title", "Price", "Ratings", "Review", "Monthly Sales"}
    missing = required - set(df_raw.columns)
    if missing:
        st.error("CSV missing required columns: " + ", ".join(sorted(missing)))
        return

    # Map & parse columns
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

    df["price"] = df["price_raw"].apply(parse_price)
    df["rating"] = df["rating_raw"].apply(parse_rating)
    df["reviews"] = df["reviews_raw"].apply(parse_reviews)
    df["sales_monthly"] = df["sales_raw"].apply(parse_sales)

    df.reset_index(inplace=True, drop=False)
    df["asin"] = df.apply(lambda r: f"SKU{int(r['index'])+1:06d}", axis=1)
    df["category"] = "Fitness"

    # Compute scores
    scored_rows = []
    for _, r in df.iterrows():
        base = r.to_dict()
        sc = compute_scores_row(base)
        base.update(sc)
        scored_rows.append(base)

    df2 = pd.DataFrame(scored_rows)

    # Search & sort
    if search_q:
        df2 = df2[
            df2["title"].str.contains(search_q, case=False, na=False)
            | df2["asin"].str.contains(search_q, case=False, na=False)
        ]

    if sort_by == "final_score":
        df2 = df2.sort_values(by="final_score", ascending=False, na_position="last")
    elif sort_by == "sales_monthly":
        df2 = df2.sort_values(by="sales_monthly", ascending=False, na_position="last")
    elif sort_by == "price":
        df2 = df2.sort_values(by="price", ascending=True, na_position="last")

    # Product table
    st.subheader("Products")
    df_display = df2[
        ["asin", "title", "price", "sales_monthly", "rating", "reviews", "final_score"]
    ].copy()

    # convert None/NaN/"None"/"nan"/"" to '-'
    df_display_clean = df_display.reset_index(drop=True).copy()

    df_display_clean = df_display_clean.applymap(
        lambda v: "-"
        if (
            v is None
            or (isinstance(v, float) and not math.isfinite(v))
            or (isinstance(v, str) and v.strip().lower() in ("none", "nan", "na", ""))
        )
        else (int(v) if isinstance(v, float) and v.is_integer() else v)
    )

    st.dataframe(df_display_clean, height=450)


    # Detail panel
    st.subheader("Product details")
    sel = st.selectbox("Select product (ASIN)", [""] + df2["asin"].tolist())

    if sel:
        p = df2[df2["asin"] == sel].iloc[0]

        left, right = st.columns([2, 1])

        with left:
            st.image(p.get("image_url"), width=320)
            st.markdown(f"### {p.get('title')}")
            st.write(f"ASIN: {p.get('asin')}  •  Category: {p.get('category')}")
            st.write(f"Price: {display_cell(p.get('price'))}")
            st.write(f"Monthly Sales: {display_cell(p.get('sales_monthly'))}")
            st.write(f"Rating: {display_cell(p.get('rating'))}")
            st.write(f"Reviews: {display_cell(p.get('reviews'))}")

            st.markdown("### Component Scores")
            st.write(f"Price score (max 20): {display_cell(p.get('score_price'))}")
            safe_progress(p.get("score_price"), WEIGHTS["price"])

            st.write(f"Review score (max 15): {display_cell(p.get('score_reviews'))}")
            safe_progress(p.get("score_reviews"), WEIGHTS["reviews"])

            st.write(f"Rating score (max 20): {display_cell(p.get('score_rating'))}")
            safe_progress(p.get("score_rating"), WEIGHTS["rating"])

            st.write(f"Monthly Sales score (max 25): {display_cell(p.get('score_sales'))}")
            safe_progress(p.get("score_sales"), WEIGHTS["sales"])

        with right:
            st.markdown("### Final Score (max 80)")
            fs = p.get("final_score")
            st.metric("Final score", display_cell(fs))
            if isinstance(fs, (int, float)) and math.isfinite(fs):
                st.progress(max(0.0, min(TOTAL_MAX_POINTS, float(fs))) / TOTAL_MAX_POINTS)
            else:
                st.write("NA")

            st.markdown("---")
            st.button("Bookmark")
            st.button("Add Note")

    st.markdown("---")
    st.info("Scoring uses your exact brackets. Missing fields show NA and are excluded from the denominator, then scaled back to a max of 80.")


if __name__ == "__main__":
    main()
