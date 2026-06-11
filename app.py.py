"""
app.py — AURORA Premium AI Concierge
Run:  streamlit run app.py
"""

import html
import os
import sqlite3

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AURORA // Premium AI Concierge",
    page_icon="⚡",
    layout="wide",
)

st.markdown("""
<style>
/* ── Base ──────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.stApp { background-color: #0d1117; color: #e6edf3; font-family: 'Inter', sans-serif; }

/* ── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] { background-color: #010409; border-right: 1px solid #21262d; }

/* ── Product card ───────────────────────────────────────────────────────── */
.product-card {
    background: linear-gradient(145deg, #161b22, #0f141c);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 22px 24px 16px;
    margin-bottom: 8px;
    transition: border-color 0.18s ease;
}
.product-card:hover { border-color: #58a6ff; }

/* ── Typography helpers ─────────────────────────────────────────────────── */
.brand-title {
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #f0f6fc;
}
.label-muted {
    color: #8b949e;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Match pill ─────────────────────────────────────────────────────────── */
.match-pill {
    background-color: rgba(56,139,253,0.12);
    color: #58a6ff;
    padding: 3px 11px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 600;
    border: 1px solid rgba(56,139,253,0.28);
    white-space: nowrap;
}

/* ── Review bubble ──────────────────────────────────────────────────────── */
.review-bubble {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
DB_FILE = "inventory.db"
CATEGORIES = [
    "electronics", "footwear", "clothing", "beauty", "home_decor",
    "fitness", "books", "automotive", "toys", "groceries",
]
FEATURE_DIM = 12          # [rating_norm, price_norm, *10-hot category]
MAX_GLOBAL_PRICE = 250_000.0
RESULTS_PER_PAGE = 5

# ── Guard: DB must exist ───────────────────────────────────────────────────────
if not os.path.exists(DB_FILE):
    st.error("**inventory.db not found.** Run `python seed_db.py` first, then reload.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# 1 · VECTOR UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _cat_onehot(category: str) -> list[float]:
    arr = [0.0] * len(CATEGORIES)
    if category in CATEGORIES:
        arr[CATEGORIES.index(category)] = 1.0
    return arr


def build_vectors_from_df(df: pd.DataFrame) -> np.ndarray:
    """Vectorise a DataFrame of products into shape (n, FEATURE_DIM)."""
    ratings = df["rating"].to_numpy(dtype=np.float32)
    prices  = (df["price"] / MAX_GLOBAL_PRICE).to_numpy(dtype=np.float32)
    cats    = np.array([_cat_onehot(c) for c in df["category"]], dtype=np.float32)
    return np.column_stack([ratings, prices, cats])


def user_vector(category: str, budget: float, min_rating: float) -> np.ndarray:
    return np.array(
        [min_rating, budget / MAX_GLOBAL_PRICE] + _cat_onehot(category),
        dtype=np.float32,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2 · DATABASE QUERY
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=120, show_spinner=False)
def _fetch_candidates_cached(
    category: str,
    budget: float,
    search: str,
    blacklist_tuple: tuple,          # hashable for cache key
) -> pd.DataFrame:
    """
    Pull matching rows from SQLite.  Blacklist filtering happens here so the
    cache key is stable for a given (category, budget, search) triple and we
    slice the blacklist out afterwards in the caller.
    """
    sql = "SELECT id, name, category, price, rating FROM products WHERE price <= ? AND category = ?"
    params: list = [budget, category]

    if search.strip():
        sql += " AND LOWER(name) LIKE ?"
        params.append(f"%{search.strip().lower()}%")

    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql_query(sql, conn, params=params)

    return df


def query_candidates(
    u_vec: np.ndarray,
    category: str,
    budget: float,
    blacklist: set,
    search: str = "",
    top_k: int = RESULTS_PER_PAGE,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return the top-k candidates ranked by cosine similarity to u_vec."""
    df = _fetch_candidates_cached(category, budget, search, tuple(sorted(blacklist)))

    if df.empty:
        return df, np.empty((0, FEATURE_DIM), dtype=np.float32)

    # Exclude dismissed items
    if blacklist:
        df = df[~df["id"].isin(blacklist)].reset_index(drop=True)

    if df.empty:
        return df, np.empty((0, FEATURE_DIM), dtype=np.float32)

    vecs = build_vectors_from_df(df)
    sims = cosine_similarity([u_vec], vecs)[0]

    actual_k = min(top_k, len(df))
    # argsort ascending → take last actual_k → highest similarity
    top_idx = np.argsort(sims)[-actual_k:][::-1]   # descending order

    return df.iloc[top_idx].copy().reset_index(drop=True), vecs[top_idx]


# ══════════════════════════════════════════════════════════════════════════════
# 3 · NEURAL RANKER
# ══════════════════════════════════════════════════════════════════════════════

class DeepRanker(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _init_ranker() -> tuple[DeepRanker, optim.Adam]:
    model = DeepRanker(FEATURE_DIM)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    return model, optimizer


# Persist model AND optimiser across reruns so Adam's moment estimates accumulate.
if "ranker_model" not in st.session_state:
    st.session_state.ranker_model, st.session_state.ranker_optim = _init_ranker()

_model: DeepRanker       = st.session_state.ranker_model
_optim: optim.Adam       = st.session_state.ranker_optim
_loss_fn = nn.BCELoss()


def backprop_step(feature_vec: np.ndarray, label: float, steps: int = 10) -> None:
    """
    In-place gradient update for a single (feature, label) pair.
    Reuses the persistent Adam optimiser so momentum carries across interactions.
    """
    X = torch.tensor(feature_vec, dtype=torch.float32).unsqueeze(0)
    y = torch.tensor([[label]], dtype=torch.float32)

    _model.train()
    for _ in range(steps):
        _optim.zero_grad()
        loss = _loss_fn(_model(X), y)
        loss.backward()
        _optim.step()
    _model.eval()


# ══════════════════════════════════════════════════════════════════════════════
# 4 · SESSION STATE DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

def _state(key: str, default):
    if key not in st.session_state:
        st.session_state[key] = default


_state("blacklist",         set())
_state("reviews",           {})    # {product_name: [{"user","stars","comment"}, …]}
_state("interaction_count", 0)


def get_reviews(product_name: str) -> list[dict]:
    if product_name not in st.session_state.reviews:
        st.session_state.reviews[product_name] = [
            {"user": "Alex M.",    "stars": 5, "comment": "Excellent craftsmanship. Exceeded expectations."},
            {"user": "S. Taylor",  "stars": 4, "comment": "Solid everyday design, sleek minimalist appearance."},
        ]
    return st.session_state.reviews[product_name]


# ══════════════════════════════════════════════════════════════════════════════
# 5 · SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("<h2 class='brand-title' style='font-size:1.4rem;'>⚡ AURORA</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p class='label-muted'>Curation Controls</p>", unsafe_allow_html=True)

_cat_display = st.sidebar.selectbox(
    "Category",
    [c.replace("_", " ").title() for c in CATEGORIES],
)
selected_category = _cat_display.lower().replace(" ", "_")

user_budget = st.sidebar.slider(
    "Max Budget (₹)", min_value=100, max_value=250_000, value=170_000, step=500,
)
user_rating = st.sidebar.slider(
    "Min Rating", min_value=1.0, max_value=5.0, value=4.0, step=0.1,
)

st.sidebar.markdown("---")

# Interaction counter gives the user a sense of how well the model is learning
interactions = st.session_state.interaction_count
st.sidebar.markdown(
    f"<p class='label-muted'>Model interactions: {interactions}</p>",
    unsafe_allow_html=True,
)

if st.sidebar.button("Reset Preferences", use_container_width=True):
    st.session_state.blacklist.clear()
    st.session_state.ranker_model, st.session_state.ranker_optim = _init_ranker()
    st.session_state.interaction_count = 0
    _model = st.session_state.ranker_model
    _optim = st.session_state.ranker_optim
    st.toast("Preferences and model reset.")
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# 6 · HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("<h1 class='brand-title' style='font-size:2.4rem; margin-bottom:4px;'>AURORA</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:#8b949e; font-size:1.05rem; margin-top:0;'>"
    "Curated items matched to your taste — and refined with every interaction.</p>",
    unsafe_allow_html=True,
)

search_input = st.text_input(
    "Search within category",
    value="",
    placeholder="Brand, model, keyword …",
    label_visibility="collapsed",
)


# ══════════════════════════════════════════════════════════════════════════════
# 7 · MAIN RESULTS
# ══════════════════════════════════════════════════════════════════════════════

u_vec = user_vector(selected_category, user_budget, user_rating)

with st.spinner("Finding your curation …"):
    candidates, vectors = query_candidates(
        u_vec, selected_category, user_budget,
        st.session_state.blacklist, search_input,
    )

if candidates.empty:
    st.markdown("---")
    st.info(
        "No items match your current filters. "
        "Try broadening your budget, lowering the minimum rating, or clearing the search field.",
        icon="🔍",
    )
    st.stop()

# Score with the neural ranker
_model.eval()
with torch.no_grad():
    scores = _model(torch.tensor(vectors, dtype=torch.float32)).numpy().flatten()

candidates = candidates.copy()
candidates["score"] = scores
ranked = candidates.sort_values("score", ascending=False).reset_index(drop=True)

st.markdown(
    f"<p class='label-muted' style='margin-top:20px;'>Suggested curations — {_cat_display}</p>",
    unsafe_allow_html=True,
)

# ── Render each product ────────────────────────────────────────────────────────
for idx, row in ranked.iterrows():
    vec     = vectors[idx]
    pct     = int(row["score"] * 100)
    safe_name = html.escape(str(row["name"]))          # XSS prevention
    safe_cat  = html.escape(str(row["category"]).upper())

    # Card header (pure HTML for layout control)
    st.markdown(f"""
    <div class='product-card'>
        <div style='display:flex; justify-content:space-between; align-items:flex-start; gap:12px;'>
            <h3 style='margin:0; color:#f0f6fc; font-weight:500; font-size:1.05rem;'>{safe_name}</h3>
            <div class='match-pill'>✦ {pct}% match</div>
        </div>
        <p style='color:#8b949e; margin:10px 0 0; font-size:0.9rem;'>
            <span style='color:#c9d1d9;'>{safe_cat}</span>
            &nbsp;·&nbsp;
            <span style='color:#58a6ff; font-weight:600;'>₹{int(row['price']):,}</span>
            &nbsp;·&nbsp;
            <span style='color:#ffd33d;'>★ {row['rating']:.1f}</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_reviews, col_actions = st.columns([3, 1])

    # ── Reviews ───────────────────────────────────────────────────────────────
    with col_reviews:
        product_reviews = get_reviews(row["name"])

        with st.expander(f"Reviews ({len(product_reviews)})"):
            for rev in product_reviews:
                stars_str = "★" * rev["stars"] + "☆" * (5 - rev["stars"])
                safe_user    = html.escape(str(rev["user"]))
                safe_comment = html.escape(str(rev["comment"]))
                st.markdown(
                    f"<div class='review-bubble'>"
                    f"<strong style='color:#c9d1d9;'>{safe_user}</strong>"
                    f"&nbsp;<span style='color:#ffd33d; font-size:0.85rem;'>{stars_str}</span><br>"
                    f"<span style='color:#8b949e; font-size:0.88rem;'>{safe_comment}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # Inline review form
            with st.form(key=f"review_form_{row['id']}", clear_on_submit=True):
                c1, c2 = st.columns([3, 1])
                u_name    = c1.text_input("Name", value="", placeholder="Your name", key=f"rn_{row['id']}")
                u_stars   = c2.slider("Stars", 1, 5, 5, key=f"rs_{row['id']}")
                u_comment = st.text_area(
                    "Comment",
                    placeholder="Share your experience with this product …",
                    key=f"rc_{row['id']}",
                )
                if st.form_submit_button("Post review"):
                    body = u_comment.strip()
                    if body:
                        author = u_name.strip() or "Anonymous"
                        st.session_state.reviews[row["name"]].insert(0, {
                            "user":    author,
                            "stars":   u_stars,
                            "comment": body,
                        })
                        st.toast("Review posted.")
                        st.rerun()
                    else:
                        st.warning("Write something before posting.")

    # ── Preference buttons ────────────────────────────────────────────────────
    with col_actions:
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        b1, b2 = st.columns(2)

        if b1.button("✨ Keep", key=f"keep_{row['id']}", use_container_width=True):
            backprop_step(vec, 1.0)
            st.session_state.interaction_count += 1
            st.toast(f"Preference saved for {row['name'][:30]}…")
            st.rerun()

        if b2.button("Dismiss", key=f"dismiss_{row['id']}", use_container_width=True):
            backprop_step(vec, 0.0)
            st.session_state.blacklist.add(row["id"])
            st.session_state.interaction_count += 1
            st.toast(f"Dismissed.")
            st.rerun()

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
