# app/main.py
# Product Discovery — MVP with YOUR custom scoring system + pagination

import math
import re
from io import StringIO

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Product Discovery — MVP", layout="wide")

# -----------------------
# CONFIG: WEIGHTS / MAX POINTS
# -----------------------
WEIGHTS = {
    "price": 20,
    "reviews": 15,
    "rating": 20,
    "sales": 25,
}
TOTAL_MAX_POINTS = sum(WEIGHTS.values())  # 80
PAGE_SIZE = 10  # items per page

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

    m = NUM_RE.search(s)
    if not m:
        return None

    raw = m.group(0).replace(",", "")
    end_pos = m.end()

    suffix = s[end_pos:].lstrip()
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
    """Extract float-like number from messy text, e.g. '4.0 out of 5 stars' -> 4.0."""
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
    """Parse price like '₹249', '1,199', '299.00' -> float or None."""
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
    """Parse rating like '4.0 out of 5 stars' -> 4.0 (clamped to 0–5)."""
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
def score_price(price):
    # PRICE (max 20)
    # <200 = 5, 200-250 = 10, 250-300 = 15, >300 = 20
    if price is None:
        return None
    try:
        p = float(price)
    except Exception:
        return None
    if not math.isfinite(p):
        return None

    if p < 200:
        return 5
    elif p <= 250:
        return 10
    elif p <= 300:
        return 15
    else:
        return 20


def score_reviews(reviews):
    # REVIEW (max 15)
    # <50 = 5, 50-500 = 10, 500-1000 = 15, >1000 = 5
    if reviews is None:
        return None
    try:
        r = float(reviews)
    except Exception:
        return None
    if not math.isfinite(r):
        return None
    r = int(r)

    if r < 50:
        return 5
    elif r <= 500:
        return 10
    elif r <= 1000:
        return 15
    else:
        return 5


def score_rating(rating):
    # RATING (max 20)
    # <3.4 = 5, 3.4-3.7 = 10, 3.7-4.0 = 15, >4.0 = 20
    if rating is None:
        return None
    try:
        r = float(rating)
    except Exception:
        return None
    if not math.isfinite(r):
        return None

    if r < 3.4:
        return 5
    elif r <= 3.7:
        return 10
    elif r <= 4.0:
        return 15
    else:
        return 20


def score_sales(sales):
    # MONTHLY SALES (max 25)
    # <100 = 5, 100-200 = 15, 200-500 = 20, 500-1000 = 25, 1000-5000 = 10, >5000 = 5
    if sales is None:
        return None
    try:
        s = float(sales)
    except Exception:
        return None
    if not math.isfinite(s):
        return None
    s = int(s)

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
    Missing fields:
      - component score = None
      - excluded from available_weight
      - final_score scaled back to TOTAL_MAX_POINTS (80)
    """
    scores = {}
    available_weight = 0
    obtained_points = 0

    sp = score_price(row.get("price"))
    scores["score_price"] = sp
    if sp is not None:
        available_weight += WEIGHTS["price"]
        obtained_points += sp

    srv = score_reviews(row.get("reviews"))
    scores["score_reviews"] = srv
    if srv is not None:
        available_weight += WEIGHTS["reviews"]
        obtained_points += srv

    srt = score_rating(row.get("rating"))
    scores["score_rating"] = srt
    if srt is not None:
        available_weight += WEIGHTS["rating"]
        obtained_points += srt

    ss = score_sales(row.get("sales_monthly"))
    scores["score_sales"] = ss
    if ss is not None:
        available_weight += WEIGHTS["sales"]
        obtained_points += ss

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
        st.write("-")
        return
    if not math.isfinite(v):
        st.write("-")
        return
    v = max(0.0, min(max_points, v))
    st.progress(v / max_points)


def format_value(val):
    """Formatter for bookmarks table (st.dataframe)."""
    if (
        val is None
        or (isinstance(val, float) and not math.isfinite(val))
        or (isinstance(val, str) and val.strip().lower() in ("none", "nan", "na", ""))
    ):
        return "-"
    if isinstance(val, float) and val.is_integer():
        return str(int(val))
    return str(val)


def render_pagination(total_items):
    """Render pagination controls top + bottom, return (start_idx, end_idx)."""
    if "current_page" not in st.session_state:
        st.session_state.current_page = 1

    current_page = st.session_state.current_page
    if total_items == 0:
        total_pages = 1
    else:
        total_pages = math.ceil(total_items / PAGE_SIZE)

    # Clamp current_page into valid range
    if current_page < 1:
        current_page = 1
    if current_page > total_pages:
        current_page = total_pages
    st.session_state.current_page = current_page

    def pager_row(suffix: str):
        col_prev, col_info, col_next = st.columns([1, 2, 1])
        with col_info:
            st.write(f"Page {current_page} / {total_pages}")
        with col_prev:
            if st.button("Prev", key=f"prev_{suffix}", disabled=current_page <= 1):
                st.session_state.current_page = max(1, current_page - 1)
        with col_next:
            if st.button("Next", key=f"next_{suffix}", disabled=current_page >= total_pages):
                st.session_state.current_page = min(total_pages, current_page + 1)

    # Top controls
    pager_row("top")

    # After possible button clicks, refresh current_page
    current_page = st.session_state.current_page
    if total_items == 0:
        total_pages = 1
    else:
        total_pages = math.ceil(total_items / PAGE_SIZE)

    start_idx = (current_page - 1) * PAGE_SIZE
    end_idx = start_idx + PAGE_SIZE

    return start_idx, end_idx, total_pages


def render_pagination_bottom(total_items, total_pages):
    """Render bottom pagination row (using current_page from session_state)."""
    current_page = st.session_state.current_page
    def pager_row(suffix: str):
        col_prev, col_info, col_next = st.columns([1, 2, 1])
        with col_info:
            st.write(f"Page {current_page} / {total_pages}")
        with col_prev:
            if st.button("Prev", key=f"prev_{suffix}", disabled=current_page <= 1):
                st.session_state.current_page = max(1, current_page - 1)
        with col_next:
            if st.button("Next", key=f"next_{suffix}", disabled=current_page >= total_pages):
                st.session_state.current_page = min(total_pages, current_page + 1)

    pager_row("bottom")


# -----------------------
# MAIN APP
# -----------------------
def main():
    st.title("Product Discovery — MVP (Custom Scoring)")
    st.write("Upload CSV with columns: **Image, Title, Price, Ratings, Review, Monthly Sales**")

    # session state for bookmarks and selected product
    if "bookmarks" not in st.session_state:
        st.session_state.bookmarks = []
    if "selected_asin" not in st.session_state:
        st.session_state.selected_asin = None
    if "current_page" not in st.session_state:
        st.session_state.current_page = 1
    if "scroll_to_details" not in st.session_state:
        st.session_state.scroll_to_details = False


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
        search_q = st.text_input("Search Title", "")
        
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

    # Search
    if search_q:
        df2 = df2[df2["title"].str.contains(search_q, case=False, na=False)]

    # Sort by:
    # 1) info completeness (how many of price/rating/reviews/sales_monthly are present)
    # 2) final_score (higher first)
    df2["info_count"] = df2[["price", "rating", "reviews", "sales_monthly"]].notna().sum(axis=1)

    df2 = df2.sort_values(
        by=["info_count", "final_score"],
        ascending=[False, False],
        na_position="last",
    )

    # -----------------------
    # Product "table" with View buttons + pagination (no ASIN shown)
    # -----------------------
    st.subheader("Products")

    total_items = len(df2)
    if total_items == 0:
        st.write("No products to show.")
        selected_product = None
    else:
        # TOP pagination
        start_idx, end_idx, total_pages = render_pagination(total_items)
        current_page = st.session_state.current_page

        # slice current page
        df_page = df2.iloc[start_idx:end_idx]

        # header row
        h_cols = st.columns([4, 1, 1, 1, 1, 1, 1])
        h_cols[0].markdown("**Title**")
        h_cols[1].markdown("**Price**")
        h_cols[2].markdown("**Monthly Sales**")
        h_cols[3].markdown("**Rating**")
        h_cols[4].markdown("**Reviews**")
        h_cols[5].markdown("**Final Score**")
        h_cols[6].markdown("**Action**")

        for _, row in df_page.iterrows():
            cols = st.columns([4, 1, 1, 1, 1, 1, 1])
            cols[0].write(display_cell(row.get("title")))
            cols[1].write(display_cell(row.get("price")))
            cols[2].write(display_cell(row.get("sales_monthly")))
            cols[3].write(display_cell(row.get("rating")))
            cols[4].write(display_cell(row.get("reviews")))
            cols[5].write(display_cell(row.get("final_score")))

            if cols[6].button("View", key=f"view_{row.get('asin')}"):
                st.session_state.selected_asin = row.get("asin")
                st.session_state.scroll_to_details = True

        # BOTTOM pagination
        render_pagination_bottom(total_items, total_pages)

    # -----------------------
    # Detail panel
    # -----------------------
    # anchor for scrolling
    st.markdown('<div id="details-anchor"></div>', unsafe_allow_html=True)
    st.subheader("Product details")

    # if a View was just clicked, scroll to this section
    if st.session_state.get("scroll_to_details"):
        st.markdown(
            """
            <script>
            var el = document.getElementById('details-anchor');
            if (el) {
                el.scrollIntoView({behavior: 'smooth', block: 'start'});
            }
            </script>
            """,
            unsafe_allow_html=True,
        )
        st.session_state.scroll_to_details = False

    if len(df2) == 0 or st.session_state.selected_asin is None:
        st.write("Click 'View' on a product above to see details.")
    else:
        matches = df2[df2["asin"] == st.session_state.selected_asin]
        if matches.empty:
            st.write("Selected product not found (maybe filtered out).")
        else:
            p = matches.iloc[0]
            sel = p.get("asin")

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
                    st.write("-")

                st.markdown("---")

                # Bookmark button
                if st.button("Bookmark"):
                    if sel not in st.session_state.bookmarks:
                        st.session_state.bookmarks.append(sel)
                        st.success("Bookmarked!")
                    else:
                        st.info("Already bookmarked.")

                # Notes placeholder (future)
                st.button("Add Note")

    # -----------------------
    # Bookmarks Section
    # -----------------------
    st.subheader("Your Bookmarks")

    if len(st.session_state.bookmarks) == 0:
        st.write("No bookmarks yet.")
    else:
        bookmarked_products = df2[df2["asin"].isin(st.session_state.bookmarks)]
        st.dataframe(
            bookmarked_products[
                ["title", "price", "sales_monthly", "rating", "reviews", "final_score"]
            ].reset_index(drop=True).style.format(format_value),
            height=300,
        )

    st.markdown("---")
    st.info("Scoring uses your exact brackets. Missing fields show '-' and are excluded from the denominator, then scaled back to a max of 80.")


if __name__ == "__main__":
    main()
