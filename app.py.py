"""
app.py — AURORA Premium AI Concierge
Streamlit application incorporating persistent neural rankers, database-driven review submission,
confetti-based shopping checkout, database inventory browser, and premium glassmorphic stylesheets.
Run:  streamlit run app.py
"""

import html
import os
import re
import sqlite3
import random
import time
import pandas as pd
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AURORA // Premium AI Concierge",
    page_icon="✨",
    layout="wide",
)

DB_FILE = "inventory.db"
MODEL_WEIGHTS_FILE = "ranker_weights.pth"
CATEGORIES = [
    "electronics", "footwear", "clothing", "beauty", "home_decor",
    "fitness", "books", "automotive", "toys", "groceries",
]
FEATURE_DIM = 16          # [rating_norm, price_norm, stock_norm, is_premium, budget_dist, rating_dist, *10-hot category]
MAX_GLOBAL_PRICE = 250_000.0
RESULTS_PER_PAGE = 5

# ── Auto-Initialization: SQLite Seeding for Streamlit Cloud ────────────────────
def check_and_seed_db():
    needs_seeding = False
    if not os.path.exists(DB_FILE):
        needs_seeding = True
    else:
        try:
            with sqlite3.connect(DB_FILE) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='products'")
                if not cursor.fetchone():
                    needs_seeding = True
                else:
                    cursor.execute("PRAGMA table_info(products)")
                    cols = [row[1] for row in cursor.fetchall()]
                    required_cols = ["brand", "item", "modifier", "stock", "description", "image_url"]
                    if not all(c in cols for c in required_cols):
                        needs_seeding = True
                
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='reviews'")
                if not cursor.fetchone():
                    needs_seeding = True
        except Exception:
            needs_seeding = True
            
    if needs_seeding:
        st.toast("⚡ Database schema outdated or missing. Re-seeding database...")
        try:
            import seed_db
            seed_db.seed_complete_database()
            st.success("Database successfully initialized/updated with 2,000 products and 6,000 reviews!")
        except Exception as e:
            st.error(f"Failed to auto-seed database: {e}")
            st.stop()

check_and_seed_db()

# ── Aesthetic Paradigm Configuration ──────────────────────────────────────────
THEMES = {
    "Nebula Amethyst": {
        "bg": "radial-gradient(circle at 50% 0%, #151030 0%, #080711 70%)",
        "primary": "#a855f7",
        "secondary": "#ec4899",
        "border": "rgba(168, 85, 247, 0.15)",
        "border_hover": "rgba(168, 85, 247, 0.6)",
        "accent": "#c084fc",
        "card_bg": "rgba(22, 20, 48, 0.45)",
        "glow_shadow": "rgba(168, 85, 247, 0.15)",
        "chart_color": "#e879f9",
        "matplotlib_border": (168/255, 85/255, 247/255, 0.15),
    },
    "Royale Gold": {
        "bg": "radial-gradient(circle at 50% 0%, #2a1f10 0%, #0a0805 70%)",
        "primary": "#d97706",
        "secondary": "#fbbf24",
        "border": "rgba(217, 119, 6, 0.15)",
        "border_hover": "rgba(251, 191, 36, 0.6)",
        "accent": "#fbbf24",
        "card_bg": "rgba(35, 28, 18, 0.45)",
        "glow_shadow": "rgba(251, 191, 36, 0.15)",
        "chart_color": "#fbbf24",
        "matplotlib_border": (217/255, 119/255, 6/255, 0.15),
    },
    "Emerald Premium": {
        "bg": "radial-gradient(circle at 50% 0%, #0a251c 0%, #040806 70%)",
        "primary": "#059669",
        "secondary": "#34d399",
        "border": "rgba(5, 150, 105, 0.15)",
        "border_hover": "rgba(52, 211, 153, 0.6)",
        "accent": "#34d399",
        "card_bg": "rgba(12, 33, 25, 0.45)",
        "glow_shadow": "rgba(52, 211, 153, 0.15)",
        "chart_color": "#34d399",
        "matplotlib_border": (5/255, 150/255, 105/255, 0.15),
    }
}

theme_choice = st.sidebar.selectbox(
    "Aesthetic Paradigm",
    list(THEMES.keys()),
    key="app_theme"
)
theme_colors = THEMES[theme_choice]

# Inject the dynamic CSS styles based on selected theme
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

:root {{
    --bg-gradient: {theme_colors["bg"]};
    --primary-glow: {theme_colors["primary"]};
    --secondary-glow: {theme_colors["secondary"]};
    --border-color: {theme_colors["border"]};
    --border-hover: {theme_colors["border_hover"]};
    --accent: {theme_colors["accent"]};
    --card-bg: {theme_colors["card_bg"]};
    --glow-shadow: {theme_colors["glow_shadow"]};
}}

/* Base Styles */
.stApp {{
    background: var(--bg-gradient) !important;
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}}

h1, h2, h3, .brand-title {{
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    letter-spacing: -0.02em;
}}

/* Sidebar Custom Styling */
[data-testid="stSidebar"] {{
    background-color: rgba(8, 7, 17, 0.85);
    border-right: 1px solid var(--border-color);
    backdrop-filter: blur(12px);
}}

/* Premium Card Layout */
.product-card {{
    background: var(--card-bg);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
}}
.product-card:hover {{
    transform: translateY(-3px);
    border-color: var(--border-hover);
    box-shadow: 0 10px 30px var(--glow-shadow);
}}

.product-image {{
    width: 100%;
    height: 180px;
    object-fit: cover;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    margin-bottom: 15px;
}}

/* Match Indicator Pill */
.match-pill {{
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.1) 100%);
    color: var(--secondary-glow);
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 700;
    border: 1px solid var(--secondary-glow);
    letter-spacing: 0.05em;
    white-space: nowrap;
}}

/* Badges */
.stock-badge-ok {{
    background-color: rgba(16, 185, 129, 0.15);
    color: #34d399;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    border: 1px solid rgba(16, 185, 129, 0.3);
}}
.stock-badge-low {{
    background-color: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    border: 1px solid rgba(245, 158, 11, 0.3);
}}
.stock-badge-out {{
    background-color: rgba(239, 68, 68, 0.15);
    color: #f87171;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    border: 1px solid rgba(239, 68, 68, 0.3);
}}

/* Reviews and Comments */
.review-box {{
    background: rgba(8, 7, 17, 0.6);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
}}

/* Utility classes */
.gradient-text {{
    background: linear-gradient(90deg, var(--primary-glow), var(--secondary-glow));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}
.label-muted {{
    color: #94a3b8;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
}}

/* Streamlit Element Overrides */
div.stButton > button {{
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid var(--border-color) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}}
div.stButton > button:hover {{
    background: var(--border-color) !important;
    border-color: var(--border-hover) !important;
    color: #ffffff !important;
    box-shadow: 0 0 15px var(--glow-shadow) !important;
    transform: translateY(-1px);
}}
div.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, var(--primary-glow) 0%, var(--secondary-glow) 100%) !important;
    border: 1px solid var(--secondary-glow) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}}
div.stButton > button[kind="primary"]:hover {{
    box-shadow: 0 0 20px var(--glow-shadow) !important;
}}

div[data-testid="stTabBar"] {{
    background: rgba(255, 255, 255, 0.02) !important;
    border-bottom: 1px solid var(--border-color) !important;
    border-radius: 8px 8px 0 0;
    padding: 0 10px;
}}
button[data-baseweb="tab"] {{
    color: #94a3b8 !important;
    font-weight: 500 !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: var(--secondary-glow) !important;
    border-bottom-color: var(--secondary-glow) !important;
}}

div[data-testid="stExpander"] {{
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
}}

div[data-baseweb="select"], div[data-baseweb="input"] {{
    background-color: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid var(--border-color) !important;
}}

@keyframes pulse {{
    0% {{ opacity: 0.6; }}
    50% {{ opacity: 1; }}
    100% {{ opacity: 0.6; }}
}}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1 · FEATURE VECTOR CONVERTER & NLP PARSER
# ══════════════════════════════════════════════════════════════════════════════

def _cat_onehot(category: str) -> list[float]:
    arr = [0.0] * len(CATEGORIES)
    if category in CATEGORIES:
        arr[CATEGORIES.index(category)] = 1.0
    return arr

def build_vectors_from_df(df: pd.DataFrame, budget: float, min_rating: float) -> np.ndarray:
    """Vectorize products into shape (n, FEATURE_DIM) using rich relational parameters."""
    ratings = df["rating"].to_numpy(dtype=np.float32)
    prices = df["price"].to_numpy(dtype=np.float32)
    stocks = df["stock"].to_numpy(dtype=np.float32)
    modifiers = df["modifier"].to_list()
    cats = df["category"].to_list()

    # 1. Normalized overall rating
    r_norm = ratings / 5.0
    # 2. Normalized price
    p_norm = prices / MAX_GLOBAL_PRICE
    # 3. Normalized stock size
    s_norm = np.minimum(stocks, 100.0) / 100.0
    # 4. Premium modifier tag indicator
    prem_list = ["Pro", "Max", "Ultra", "Elite", "Signature", "Advanced", "Premium", "Select"]
    is_prem = np.array([1.0 if m in prem_list else 0.0 for m in modifiers], dtype=np.float32)
    # 5. Price deviation against user budget constraint
    b_dist = (budget - prices) / (budget + 1e-5)
    # 6. Rating distance from threshold
    r_dist = (ratings - min_rating) / 5.0
    # 7. One-hot categories array
    cat_vecs = np.array([_cat_onehot(c) for c in cats], dtype=np.float32)

    return np.column_stack([
        r_norm, p_norm, s_norm, is_prem, b_dist, r_dist, cat_vecs
    ])

def user_vector(category: str, budget: float, min_rating: float) -> np.ndarray:
    """Represents the ideal utility profile vectors desired by the user."""
    ideal_rating_norm = 1.0
    ideal_price_norm = budget / MAX_GLOBAL_PRICE
    ideal_stock_norm = 0.5
    ideal_prem = 1.0
    ideal_b_dist = 0.5
    ideal_r_dist = (5.0 - min_rating) / 5.0
    ideal_cat = _cat_onehot(category)
    return np.array(
        [ideal_rating_norm, ideal_price_norm, ideal_stock_norm, ideal_prem, ideal_b_dist, ideal_r_dist] + ideal_cat,
        dtype=np.float32,
    )

def parse_concierge_query(query: str) -> tuple[str, bool]:
    """Parse dialogue search strings to configure widgets automatically."""
    q = query.lower()
    
    # 1. Parse Category
    found_category = None
    category_mappings = {
        "electronics": ["electronic", "tech", "laptop", "phone", "tv", "earbud", "headphones", "monitor", "ssd", "router"],
        "footwear": ["footwear", "shoe", "sneaker", "boot", "clog", "sandals", "slipper"],
        "clothing": ["clothing", "shirt", "jeans", "jacket", "hoodie", "tee", "sweaters", "chino", "polo", "wear"],
        "beauty": ["beauty", "skin", "cream", "essence", "serum", "spf", "sunscreen", "shampoo", "lipstick", "fragrance", "perfume"],
        "home_decor": ["decor", "home", "clock", "candle", "lamp", "vase", "furniture", "rug", "blanket", "chair"],
        "fitness": ["fitness", "gym", "workout", "dumbbells", "yoga", "tracker", "gps", "bike", "rowing", "whey", "protein"],
        "books": ["book", "read", "guide", "handbook", "blueprint", "algorithms", "learning"],
        "automotive": ["automotive", "car", "dashcam", "wiper", "tyre", "pressure", "vehicle", "inflator"],
        "toys": ["toy", "game", "lego", "drone", "blaster", "puzzle", "doll", "play"],
        "groceries": ["groceries", "food", "coffee", "tea", "oil", "honey", "chocolate", "oats", "almond", "spread"]
    }
    
    for cat, synonyms in category_mappings.items():
        if cat in q or any(syn in q for syn in synonyms):
            found_category = cat
            break
            
    # 2. Parse Budget Limits
    found_budget = None
    k_match = re.search(r'(?:under|below|less than|max|budget of|budget)\s*(?:₹|rs\.?)?\s*(\d+(?:\.\d+)?)\s*(k|thousand)', q)
    lakh_match = re.search(r'(?:under|below|less than|max|budget of|budget)\s*(?:₹|rs\.?)?\s*(\d+(?:\.\d+)?)\s*(lakh|lacs|lakhs)', q)
    num_match = re.search(r'(?:under|below|less than|max|budget of|budget)\s*(?:₹|rs\.?)?\s*(\d{3,6})', q)
    
    if lakh_match:
        found_budget = float(lakh_match.group(1)) * 100_000
    elif k_match:
        found_budget = float(k_match.group(1)) * 1000
    elif num_match:
        found_budget = float(num_match.group(1))
        
    # 3. Parse Ratings Limits
    found_rating = None
    rating_match = re.search(r'(?:at least|above|rating of|min|minimum)\s*(\d(?:\.\d+)?)\s*(?:stars|star|★)?', q)
    plus_match = re.search(r'(\d(?:\.\d+)?)\s*\+\s*(?:stars|star|★)', q)
    stars_only = re.search(r'(\d(?:\.\d+)?)\s*(?:stars|star|★)', q)
    
    if rating_match:
        found_rating = float(rating_match.group(1))
    elif plus_match:
        found_rating = float(plus_match.group(1))
    elif stars_only:
        found_rating = float(stars_only.group(1))

    # 4. Extract Brand Keywords
    all_brands = ["apple", "samsung", "sony", "dell", "hp", "asus", "lenovo", "logitech", "lg", "bose", "nike", "adidas", "puma", "levis", "dyson", "lego", "philips"]
    found_brand = None
    for brand in all_brands:
        if brand in q:
            found_brand = brand
            break

    changes = []
    if found_category:
        cat_disp = found_category.replace("_", " ").title()
        st.session_state["category_filter"] = cat_disp
        changes.append(f"Category set to **{cat_disp}**")
    if found_budget:
        budget_val = int(np.clip(found_budget, 100, 250000))
        st.session_state["budget_filter"] = budget_val
        changes.append(f"Budget limit adjusted to **₹{budget_val:,}**")
    if found_rating:
        rating_val = float(np.clip(found_rating, 1.0, 5.0))
        st.session_state["rating_filter"] = rating_val
        changes.append(f"Minimum rating set to **{rating_val:.1f} ★**")
    if found_brand:
        st.session_state["search_input_val"] = found_brand
        changes.append(f"Search keyword set to **'{found_brand.title()}'**")
        
    if changes:
        msg = f"✦ **Aurora Concierge**: *'Welcome back, Client. I have processed your instruction: {', '.join(changes)}. Let me know if you would like to refine this luxury curation further.'*"
        return msg, True
    else:
        return "✦ **Aurora Concierge**: *'I received your request, but could not parse specific parameters. Try specifying a category (e.g. footwear, beauty), a budget (e.g. under 1.5 lakhs), or a rating threshold (e.g. 4.5+ stars).'*", False

# ══════════════════════════════════════════════════════════════════════════════
# 2 · CACHED DATABASE READ OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=60, show_spinner=False)
def _fetch_candidates_cached(
    category: str,
    budget: float,
    search: str,
    blacklist_tuple: tuple,
) -> pd.DataFrame:
    """Query base candidates from products matching budget constraints."""
    sql = "SELECT id, name, category, price, rating, brand, item, modifier, stock, description, image_url FROM products WHERE price <= ? AND category = ?"
    params = [budget, category]

    if search.strip():
        sql += " AND (LOWER(name) LIKE ? OR LOWER(brand) LIKE ?)"
        term = f"%{search.strip().lower()}%"
        params.extend([term, term])

    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql_query(sql, conn, params=params)

    return df

def query_candidates(
    u_vec: np.ndarray,
    category: str,
    budget: float,
    blacklist: set,
    search: str = "",
    top_k: int = 25,  # Fetch a larger candidate pool to neural rank
) -> tuple[pd.DataFrame, np.ndarray]:
    df = _fetch_candidates_cached(category, budget, search, tuple(sorted(blacklist)))

    if df.empty:
        return df, np.empty((0, FEATURE_DIM), dtype=np.float32)

    if blacklist:
        df = df[~df["id"].isin(blacklist)].reset_index(drop=True)

    if df.empty:
        return df, np.empty((0, FEATURE_DIM), dtype=np.float32)

    vecs = build_vectors_from_df(df, budget, u_vec[5] * 5.0)  # extract min_rating back
    sims = cosine_similarity([u_vec], vecs)[0]

    # Rank indices based on similarity first to slice top-k
    top_idx = np.argsort(sims)[-top_k:][::-1]
    return df.iloc[top_idx].copy().reset_index(drop=True), vecs[top_idx]

# ══════════════════════════════════════════════════════════════════════════════
# 3 · NEURAL RANKER SYSTEM
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
    optimizer = optim.Adam(model.parameters(), lr=0.008, weight_decay=1e-4)

    # Load persistent weights if present
    if os.path.exists(MODEL_WEIGHTS_FILE):
        try:
            state = torch.load(MODEL_WEIGHTS_FILE)
            model.load_state_dict(state.get("model_state"))
            optimizer.load_state_dict(state.get("optimizer_state"))
        except Exception:
            pass # Suppress loading failure to prevent app stopping

    return model, optimizer

# Initialize state persistence
if "ranker_model" not in st.session_state:
    st.session_state.ranker_model, st.session_state.ranker_optim = _init_ranker()

_model: DeepRanker = st.session_state.ranker_model
_optim: optim.Adam = st.session_state.ranker_optim
_loss_fn = nn.BCELoss()

def save_ranker_weights() -> None:
    """Save weights to disk for persistence across runs."""
    torch.save({
        "model_state": _model.state_dict(),
        "optimizer_state": _optim.state_dict()
    }, MODEL_WEIGHTS_FILE)

def backprop_step(feature_vec: np.ndarray, label: float, steps: int = 15) -> float:
    """Adam gradient updates for neural ranks."""
    X = torch.tensor(feature_vec, dtype=torch.float32).unsqueeze(0)
    y = torch.tensor([[label]], dtype=torch.float32)

    _model.train()
    avg_loss = 0.0
    for _ in range(steps):
        _optim.zero_grad()
        loss = _loss_fn(_model(X), y)
        loss.backward()
        _optim.step()
        avg_loss += loss.item()
    _model.eval()

    save_ranker_weights()

    # Track loss history in session state
    if "loss_history" not in st.session_state:
        st.session_state.loss_history = []
    st.session_state.loss_history.append(avg_loss / steps)

    return avg_loss / steps

# ══════════════════════════════════════════════════════════════════════════════
# 4 · SESSION STATE AND MOCK TRANSACTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _state(key: str, default):
    if key not in st.session_state:
        st.session_state[key] = default

# Syncable Widgets
_state("category_filter", "Electronics")
_state("budget_filter", 180000)
_state("rating_filter", 4.2)
_state("search_input_val", "")

_state("blacklist", set())
_state("interaction_count", 0)
_state("loss_history", [])
_state("shopping_cart", {})       # {product_id: {"name": str, "price": float, "qty": int}}
_state("wishlist", set())          # {product_ids}
_state("comparison_pool", [])      # [product_ids]
_state("manual_weights_enabled", False)
_state("manual_weights", {})
_state("concierge_dialogue_msg", None)
_state("order_placed", False)
_state("order_details", None)

# ══════════════════════════════════════════════════════════════════════════════
# 5 · SIDEBAR CONTROLS
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("<h2 class='brand-title' style='color:var(--primary-glow); font-size:1.6rem;'>⚡ AURORA</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p class='label-muted'>Luxury AI Concierge System</p>", unsafe_allow_html=True)

selected_cat_disp = st.sidebar.selectbox(
    "Curated Category",
    [c.replace("_", " ").title() for c in CATEGORIES],
    key="category_filter"
)
selected_category = selected_cat_disp.lower().replace(" ", "_")

user_budget = st.sidebar.slider(
    "Target Budget Limit (₹)", min_value=100, max_value=250_000, step=500,
    key="budget_filter"
)
user_rating = st.sidebar.slider(
    "Minimum Client Rating", min_value=1.0, max_value=5.0, step=0.1,
    key="rating_filter"
)

st.sidebar.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

# Performance metrics
st.sidebar.markdown("<p class='label-muted'>AI Engine Status</p>", unsafe_allow_html=True)
c_logs1, c_logs2 = st.sidebar.columns(2)
c_logs1.metric("Interactions", st.session_state.interaction_count)
current_loss = f"{st.session_state.loss_history[-1]:.4f}" if st.session_state.loss_history else "N/A"
c_logs2.metric("Ranker Loss", current_loss)

if st.sidebar.button("Reset AI Core Preferences", use_container_width=True):
    # Wipe weights
    if os.path.exists(MODEL_WEIGHTS_FILE):
        try:
            os.remove(MODEL_WEIGHTS_FILE)
        except Exception:
            pass
    st.session_state.blacklist.clear()
    st.session_state.interaction_count = 0
    st.session_state.loss_history = []
    st.session_state.shopping_cart.clear()
    st.session_state.wishlist.clear()
    st.session_state.comparison_pool.clear()
    st.session_state.manual_weights_enabled = False
    st.session_state.manual_weights.clear()
    st.session_state.concierge_dialogue_msg = None
    st.session_state.order_placed = False
    st.session_state.order_details = None
    st.session_state.category_filter = "Electronics"
    st.session_state.budget_filter = 180000
    st.session_state.rating_filter = 4.2
    st.session_state.search_input_val = ""

    st.session_state.ranker_model, st.session_state.ranker_optim = _init_ranker()
    st.toast("Model weights and client memory cleared successfully.", icon="🧹")
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# 6 · HEADER AND SEARCH PANEL
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("<h1 class='brand-title' style='font-size:2.8rem; margin-bottom: 2px;'><span class='gradient-text'>AURORA</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#94a3b8; font-size:1.1rem; margin-top:0;'>High-fidelity recommendation engine, backpropagating choices directly in the browser.</p>", unsafe_allow_html=True)

search_input = st.text_input(
    "Search database keyword...",
    placeholder="Search by brand name, model modifiers, or product keywords...",
    label_visibility="collapsed",
    key="search_input_val"
)

# ══════════════════════════════════════════════════════════════════════════════
# 7 · INTERACTIVE MODALS
# ══════════════════════════════════════════════════════════════════════════════

@st.dialog("Product Specification & Sentiment Profile", width="large")
def show_product_details(row, vec):
    safe_name = html.escape(str(row["name"]))
    st.markdown(f"### {safe_name}")
    st.markdown(f"**Category:** {row['category'].upper()} | **Price:** ₹{int(row['price']):,} | **Rating:** ★ {row['rating']:.1f}")
    
    col_img, col_metrics = st.columns([1, 1])
    with col_img:
        st.image(row['image_url'], use_container_width=True, caption=safe_name)
        st.write(f"*{html.escape(row['description'])}*")
        
    with col_metrics:
        st.markdown("<p class='label-muted'>Client Sentiment Scan</p>", unsafe_allow_html=True)
        # Fetch reviews
        with sqlite3.connect(DB_FILE) as conn:
            db_reviews = pd.read_sql_query(
                "SELECT user, stars, comment FROM reviews WHERE product_id = ? ORDER BY id DESC",
                conn, params=[int(row["id"])]
            )
        
        # Analyze Sentiment
        pos_words = {"amazing", "love", "perfect", "excellent", "stunning", "great", "comfortable", "best", "glowing", "highly", "top-notch", "impressed", "worth", "smooth", "luxurious", "exquisite", "pure", "fresh", "classic", "premium"}
        neg_words = {"tight", "expensive", "shrank", "medicinal", "crease", "shorter", "dense", "overpriced", "slow", "heavy", "dull", "missing", "confusing", "tricky", "bad", "charge", "fail"}
        
        total_pos = 0
        total_words = 0
        for _, r_row in db_reviews.iterrows():
            comment = str(r_row['comment']).lower()
            words = comment.split()
            p_count = sum(1 for w in words if w in pos_words)
            n_count = sum(1 for w in words if w in neg_words)
            total_pos += p_count
            total_words += (p_count + n_count)
            
        if total_words > 0:
            sentiment_pct = int((total_pos / total_words) * 100)
        else:
            sentiment_pct = int((row['rating'] / 5.0) * 100)
            
        st.markdown(f"**✦ {sentiment_pct}% POSITIVE CLIENT SENTIMENT**")
        st.progress(sentiment_pct / 100.0)
        
        # features list
        st.markdown("<p class='label-muted'>Neural Vector Analysis</p>", unsafe_allow_html=True)
        st.write(f"- **Overall Quality Score**: {vec[0]*5.0:.1f} / 5.0")
        st.write(f"- **Price Dimension (Normalized)**: {vec[1]:.3f}")
        st.write(f"- **Stock Volume (Density)**: {vec[2]*100:.0f}%")
        st.write(f"- **Bespoke Modifiers**: {'Yes' if vec[3] > 0 else 'No'}")
        st.write(f"- **Target Budget Dev**: {vec[4]:.3f}")
        st.write(f"- **Rating Delta (Threshold)**: {vec[5]*5.0:.1f}")
        
    st.markdown("---")
    st.markdown("#### Customer Reviews")
    for _, r_row in db_reviews.iterrows():
        stars_str = "★" * int(r_row["stars"]) + "☆" * (5 - int(r_row["stars"]))
        st.markdown(f"""
        <div class='review-box'>
            <div style='display: flex; justify-content: space-between;'>
                <strong>{html.escape(r_row['user'])}</strong>
                <span style='color:#fbbf24;'>{stars_str}</span>
            </div>
            <p style='margin: 5px 0 0 0; color:#94a3b8; font-size:0.85rem;'>{html.escape(r_row['comment'])}</p>
        </div>
        """, unsafe_allow_html=True)
        
    # Write a review inside dialog
    with st.form(key=f"review_modal_{row['id']}", clear_on_submit=True):
        st.markdown("<p style='font-size:0.85rem; font-weight:bold; margin-bottom:8px;'>Write a Review</p>", unsafe_allow_html=True)
        c_n, c_s = st.columns([3, 1])
        rev_name = c_n.text_input("Name", placeholder="Anonymous User", key=f"mn_{row['id']}")
        rev_stars = c_s.slider("Stars", 1, 5, 5, key=f"ms_{row['id']}")
        rev_comment = st.text_area("Review Comment", placeholder="Describe your experience...", key=f"mc_{row['id']}", height=80)
        
        if st.form_submit_button("Submit Client Review"):
            if rev_comment.strip():
                author = rev_name.strip() or "Anonymous User"
                with sqlite3.connect(DB_FILE) as conn:
                    conn.execute(
                        "INSERT INTO reviews (product_id, user, stars, comment) VALUES (?, ?, ?, ?)",
                        (int(row["id"]), author, int(rev_stars), rev_comment.strip())
                    )
                    conn.commit()
                st.toast("Review recorded in database.", icon="📝")
                st.rerun()
            else:
                st.warning("Please include comments to submit feedback.")

# ══════════════════════════════════════════════════════════════════════════════
# 8 · TABS ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

t_curation, t_analytics, t_commerce, t_db, t_exclusions = st.tabs([
    "⚡ Personal Curation",
    "📊 AI Insights & Loss Curves",
    "🛒 Cart & Wishlist Checkouts",
    "🗃️ Complete Database Browser",
    "🚫 Dismissed Exclusions"
])

# ── 8.1 TAB: PERSONAL CURATION ───────────────────────────────────────────────
with t_curation:
    # NLP dialogue concierge bar
    st.markdown("<p class='label-muted' style='margin-top:10px;'>Ask Aurora Premium Concierge</p>", unsafe_allow_html=True)
    c_dialogue, c_dialogue_btn = st.columns([6, 1])
    dialogue_input = c_dialogue.text_input(
        "Ask Aurora Concierge...",
        placeholder="E.g., 'Find me clothing under 5000 with at least 4.5 stars' or 'Show me books about data'",
        label_visibility="collapsed",
        key="concierge_input"
    )
    if c_dialogue_btn.button("Send", use_container_width=True):
        if dialogue_input.strip():
            msg, matched = parse_concierge_query(dialogue_input)
            st.session_state["concierge_dialogue_msg"] = msg
            st.rerun()

    # Display dialogue response
    if st.session_state.concierge_dialogue_msg:
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.02); border: 1px solid var(--border-color); border-radius: 12px; padding: 16px; margin-bottom: 20px;">
            {st.session_state.concierge_dialogue_msg}
        </div>
        """, unsafe_allow_html=True)

    # Product Comparison Canvas
    if st.session_state.comparison_pool:
        with st.expander("⇄ Product Comparison Grid", expanded=True):
            comp_ids = st.session_state.comparison_pool
            sql = f"SELECT id, name, category, price, rating, brand, item, modifier, stock, description, image_url FROM products WHERE id IN ({','.join(['?']*len(comp_ids))})"
            with sqlite3.connect(DB_FILE) as conn:
                comp_df = pd.read_sql_query(sql, conn, params=comp_ids)
                
            if not comp_df.empty:
                comp_vecs = build_vectors_from_df(comp_df, user_budget, user_rating)
                
                # Check manual weight overlay
                if st.session_state.get("manual_weights_enabled", False):
                    w_dict = st.session_state.manual_weights
                    mw = np.array([
                        w_dict.get("rating", 1.5),
                        w_dict.get("price", -1.0),
                        w_dict.get("stock", 0.5),
                        w_dict.get("premium", 1.0),
                        w_dict.get("budget_dist", 1.2),
                        w_dict.get("rating_dist", 0.8)
                    ], dtype=np.float32)
                    raw_scores = np.dot(comp_vecs[:, :6], mw)
                    comp_scores = 1.0 / (1.0 + np.exp(-raw_scores))
                else:
                    _model.eval()
                    with torch.no_grad():
                        comp_scores = _model(torch.tensor(comp_vecs, dtype=torch.float32)).numpy().flatten()
                
                comp_df["score"] = comp_scores
                
                cols = st.columns(len(comp_df) + 1)
                with cols[0]:
                    st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)
                    st.markdown("<p style='font-size:0.9rem; margin-bottom:12px;'><strong>Price</strong></p>", unsafe_allow_html=True)
                    st.markdown("<p style='font-size:0.9rem; margin-bottom:12px;'><strong>Client Rating</strong></p>", unsafe_allow_html=True)
                    st.markdown("<p style='font-size:0.9rem; margin-bottom:12px;'><strong>Neural Match</strong></p>", unsafe_allow_html=True)
                    st.markdown("<p style='font-size:0.9rem; margin-bottom:12px;'><strong>Stock Level</strong></p>", unsafe_allow_html=True)
                    st.markdown("<p style='font-size:0.9rem;'><strong>Modifier</strong></p>", unsafe_allow_html=True)
                    
                for col_idx, (_, c_row) in enumerate(comp_df.iterrows(), start=1):
                    with cols[col_idx]:
                        st.markdown(f"""
                        <div style='text-align: center; margin-bottom:10px;'>
                            <img src="{c_row['image_url']}" style="width:70px; height:70px; object-fit:cover; border-radius:8px; border:1px solid rgba(255,255,255,0.1); margin-bottom:5px;">
                            <div style='font-weight:600; font-size:0.9rem; text-overflow: ellipsis; overflow: hidden; white-space: nowrap;'>{html.escape(c_row['name'])}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"<p style='font-size:0.9rem; margin-bottom:12px;'>₹{int(c_row['price']):,}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='font-size:0.9rem; margin-bottom:12px;'>★ {c_row['rating']:.1f}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='font-size:0.9rem; margin-bottom:12px; color:var(--secondary-glow); font-weight:700;'>✦ {int(c_row['score']*100)}%</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='font-size:0.9rem; margin-bottom:12px;'>{c_row['stock']} left</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='font-size:0.9rem;'>{c_row['modifier']}</p>", unsafe_allow_html=True)
                        
                        if st.button("Remove", key=f"rem_comp_{c_row['id']}", use_container_width=True):
                            st.session_state.comparison_pool.remove(c_row["id"])
                            st.rerun()
                            
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                if st.button("Clear Comparison Grid", use_container_width=True):
                    st.session_state.comparison_pool.clear()
                    st.rerun()

    # Query matching candidate products
    u_vec = user_vector(selected_category, user_budget, user_rating)
    with st.spinner("Analyzing matrices and scoring vectors..."):
        candidates, vectors = query_candidates(
            u_vec, selected_category, user_budget,
            st.session_state.blacklist, search_input,
        )

    if candidates.empty:
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        st.info("We couldn't find items matching your criteria. Try sliding up your budget or relaxing the ratings limit.", icon="🔍")
    else:
        # Score candidates
        if st.session_state.get("manual_weights_enabled", False):
            w_dict = st.session_state.manual_weights
            mw = np.array([
                w_dict.get("rating", 1.5),
                w_dict.get("price", -1.0),
                w_dict.get("stock", 0.5),
                w_dict.get("premium", 1.0),
                w_dict.get("budget_dist", 1.2),
                w_dict.get("rating_dist", 0.8)
            ], dtype=np.float32)
            raw_scores = np.dot(vectors[:, :6], mw)
            scores = 1.0 / (1.0 + np.exp(-raw_scores))
        else:
            _model.eval()
            with torch.no_grad():
                scores = _model(torch.tensor(vectors, dtype=torch.float32)).numpy().flatten()

        candidates = candidates.copy()
        candidates["score"] = scores
        ranked = candidates.sort_values("score", ascending=False).reset_index(drop=True)

        st.markdown(f"<p class='label-muted' style='margin-top:20px;'>Deep Curation Pipeline — {selected_cat_disp} Matches</p>", unsafe_allow_html=True)

        for idx, row in ranked.head(RESULTS_PER_PAGE).iterrows():
            vec = vectors[idx]
            match_pct = int(row["score"] * 100)
            safe_name = html.escape(str(row["name"]))
            safe_cat = html.escape(str(row["category"]).upper())
            safe_desc = html.escape(str(row["description"]))

            stock_val = row["stock"]
            if stock_val == 0:
                stock_html = "<span class='stock-badge-out'>OUT OF STOCK</span>"
            elif stock_val < 10:
                stock_html = f"<span class='stock-badge-low'>ONLY {stock_val} LEFT</span>"
            else:
                stock_html = f"<span class='stock-badge-ok'>IN STOCK ({stock_val})</span>"

            st.markdown(f"""
            <div class='product-card'>
                <div style='display: flex; gap: 24px; flex-wrap: wrap;'>
                    <div style='flex: 1; min-width: 200px; max-width: 240px;'>
                        <img src="{row['image_url']}" class='product-image' alt='{safe_name}'>
                    </div>
                    <div style='flex: 3; min-width: 300px; display: flex; flex-direction: column; justify-content: space-between;'>
                        <div>
                            <div style='display: flex; justify-content: space-between; align-items: flex-start; gap: 10px;'>
                                <h3 style='margin: 0; color: #f8fafc; font-size: 1.3rem; font-weight: 600;'>{safe_name}</h3>
                                <div class='match-pill'>✦ {match_pct}% MATCH</div>
                            </div>
                            <div style='display: flex; align-items: center; gap: 12px; margin-top: 8px;'>
                                <span style='font-size: 0.8rem; font-weight: bold; color: var(--accent); letter-spacing: 0.05em;'>{safe_cat}</span>
                                &middot;
                                <span style='color: #38bdf8; font-weight: 700; font-size: 1.1rem;'>₹{int(row['price']):,}</span>
                                &middot;
                                <span style='color: #fbbf24; font-weight: 600;'>★ {row['rating']:.1f}</span>
                                &middot;
                                {stock_html}
                            </div>
                            <p style='color: #94a3b8; font-size: 0.92rem; line-height: 1.6; margin-top: 12px; margin-bottom: 0;'>{safe_desc}</p>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_details, col_commerce, col_decision = st.columns([4, 2, 2])

            with col_details:
                st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
                if st.button("✦ View Details & Reviews", key=f"details_{row['id']}", use_container_width=True):
                    show_product_details(row, vec)

            with col_commerce:
                st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
                # Comparison toggle
                in_comp = row["id"] in st.session_state.comparison_pool
                comp_label = "⇄ Compare (Added)" if in_comp else "⇄ Compare Item"
                if st.button(comp_label, key=f"comp_{row['id']}", use_container_width=True):
                    if in_comp:
                        st.session_state.comparison_pool.remove(row["id"])
                        st.toast("Removed from comparison list.", icon="🔄")
                    else:
                        if len(st.session_state.comparison_pool) >= 3:
                            st.warning("You can compare up to 3 items at a time.")
                        else:
                            st.session_state.comparison_pool.append(row["id"])
                            st.toast("Added to comparison list.", icon="⇄")
                    st.rerun()

                # Add to Cart Actions
                if stock_val > 0:
                    if st.button("🛒 Add to Cart", key=f"cart_act_{row['id']}", use_container_width=True):
                        cart = st.session_state.shopping_cart
                        p_id = str(row["id"])
                        if p_id in cart:
                            cart[p_id]["qty"] += 1
                        else:
                            cart[p_id] = {"name": row["name"], "price": float(row["price"]), "qty": 1}
                        st.toast(f"Added {row['name']} to shopping cart.", icon="🛍️")
                else:
                    st.button("🚫 Out of Stock", disabled=True, key=f"cart_disabled_{row['id']}", use_container_width=True)

            with col_decision:
                st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
                # Keep Preference
                if st.button("✨ Keep Match", key=f"keep_{row['id']}", use_container_width=True):
                    backprop_step(vec, 1.0)
                    st.session_state.interaction_count += 1
                    st.toast("Network reinforced to rank similar properties higher.", icon="🚀")
                    st.rerun()

                # Dismiss Preference
                if st.button("Dismiss", key=f"dismiss_{row['id']}", use_container_width=True):
                    backprop_step(vec, 0.0)
                    st.session_state.blacklist.add(row["id"])
                    st.session_state.interaction_count += 1
                    st.toast("Item excluded. Weight vector adjusted.", icon="🧹")
                    st.rerun()

            st.markdown("<hr style='border-color:var(--border-color); margin:15px 0;'>", unsafe_allow_html=True)

# ── 8.2 TAB: AI INSIGHTS & LOSS CURVES ────────────────────────────────────────
with t_analytics:
    st.markdown("<h3 style='margin-top: 10px;'>Neural Ranker Diagnostic Panel</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8;'>Understanding how client choices affect the neural parameters in real-time.</p>", unsafe_allow_html=True)

    c_an1, c_an2 = st.columns(2)

    with c_an1:
        st.markdown(f"<h4 style='text-align: center; color: var(--accent);'>Training Loss Log Curve</h4>", unsafe_allow_html=True)
        if st.session_state.loss_history:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            
            # Match theme backgrounds
            bg_color = '#080711' if theme_choice == 'Nebula Amethyst' else ('#0a0805' if theme_choice == 'Royale Gold' else '#040806')
            panel_color = '#0f0d22' if theme_choice == 'Nebula Amethyst' else ('#14100b' if theme_choice == 'Royale Gold' else '#08120e')
            
            fig.patch.set_facecolor(bg_color)
            ax.set_facecolor(panel_color)
            ax.plot(st.session_state.loss_history, color=theme_colors["chart_color"], linewidth=2.5, marker='o', markersize=4)
            ax.set_title("Binary Cross-Entropy Loss Curve", color='#f8fafc', fontsize=10)
            ax.set_xlabel("Interaction Iterations", color='#94a3b8', fontsize=8)
            ax.set_ylabel("Loss Magnitude", color='#94a3b8', fontsize=8)
            ax.tick_params(colors='#94a3b8', labelsize=8)
            ax.grid(color=(1.0, 1.0, 1.0, 0.05), linestyle='--')
            for spine in ax.spines.values():
                spine.set_color(theme_colors["matplotlib_border"])
            st.pyplot(fig)
        else:
            st.info("No interactions recorded yet. Interact with product feeds (Keep / Dismiss) to build optimization analytics.")

    with c_an2:
        st.markdown(f"<h4 style='text-align: center; color: var(--secondary-glow);'>Model Weight Importance Matrix</h4>", unsafe_allow_html=True)
        try:
            weights = _model.net[0].weight.data.numpy()
            feature_importance = np.abs(weights).mean(axis=0)

            feature_labels = [
                "Overall Rating", "Price Scaled", "Stock Limit", "Is Premium Mode",
                "Budget Gap", "Rating Delta"
            ] + [f"Cat: {c.title().replace('_', ' ')}" for c in CATEGORIES]

            sorted_idx = np.argsort(feature_importance)[::-1][:8]
            display_importance = feature_importance[sorted_idx]
            display_labels = [feature_labels[i] for i in sorted_idx]

            fig, ax = plt.subplots(figsize=(6, 3.5))
            
            bg_color = '#080711' if theme_choice == 'Nebula Amethyst' else ('#0a0805' if theme_choice == 'Royale Gold' else '#040806')
            panel_color = '#0f0d22' if theme_choice == 'Nebula Amethyst' else ('#14100b' if theme_choice == 'Royale Gold' else '#08120e')
            
            fig.patch.set_facecolor(bg_color)
            ax.set_facecolor(panel_color)
            y_pos = np.arange(len(display_labels))
            ax.barh(y_pos, display_importance, align='center', color=theme_colors["chart_color"], edgecolor=theme_colors["secondary"], alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(display_labels, color='#f8fafc', fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel('Mean Absolute Connection Weight', color='#94a3b8', fontsize=8)
            ax.tick_params(colors='#94a3b8', labelsize=8)
            ax.grid(color=(1.0, 1.0, 1.0, 0.05), linestyle='--')
            for spine in ax.spines.values():
                spine.set_color(theme_colors["matplotlib_border"])
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Failed to render neural matrix plots: {e}")

    # Neural Telemetry overrides & Inspector
    st.markdown("---")
    st.markdown("### 🛠️ Manual Telemetry Overrides & Neural Flows")
    manual_enabled = st.checkbox("Enable Manual Neural Weight Injection", value=st.session_state.get("manual_weights_enabled", False), key="manual_weights_enabled_toggle")

    if manual_enabled:
        st.session_state.manual_weights_enabled = True
        st.info("Direct synaptic override active. The PyTorch ranker weights are bypassed in favor of these custom parameters.")
        
        c_w1, c_w2, c_w3 = st.columns(3)
        w_rating = c_w1.slider("Rating Weight", -5.0, 5.0, 1.5, 0.1)
        w_price = c_w2.slider("Price Weight (Lower is better)", -5.0, 5.0, -1.0, 0.1)
        w_stock = c_w3.slider("Stock Weight", -5.0, 5.0, 0.5, 0.1)
        
        c_w4, c_w5, c_w6 = st.columns(3)
        w_premium = c_w4.slider("Premium Modifiers Weight", -5.0, 5.0, 1.0, 0.1)
        w_budget_dist = c_w5.slider("Budget Deviation Weight", -5.0, 5.0, 1.2, 0.1)
        w_rating_dist = c_w6.slider("Rating Delta Weight", -5.0, 5.0, 0.8, 0.1)
        
        st.session_state.manual_weights = {
            "rating": w_rating,
            "price": w_price,
            "stock": w_stock,
            "premium": w_premium,
            "budget_dist": w_budget_dist,
            "rating_dist": w_rating_dist
        }
    else:
        st.session_state.manual_weights_enabled = False

    # Pathway inspector
    if not candidates.empty:
        st.markdown("#### Synaptic Activation Inspector")
        inspect_item_name = st.selectbox("Select Product to Inspect Pathway", candidates["name"].tolist())
        inspect_idx = candidates[candidates["name"] == inspect_item_name].index[0]
        inspect_row = candidates.iloc[inspect_idx]
        inspect_vec = vectors[inspect_idx]
        
        c_f1, c_f2 = st.columns(2)
        with c_f1:
            st.markdown(f"""
            **Computed Vector Dimensions:**
            - **Quality Stars Factor** ({inspect_vec[0]:.2f})
            - **Price Scaling Factor** ({inspect_vec[1]:.2f})
            - **Stock Allocation Factor** ({inspect_vec[2]:.2f})
            - **Bespoke Signature Mod** ({inspect_vec[3]:.2f})
            - **Budget Constraint Dev** ({inspect_vec[4]:.2f})
            - **Rating Delta Margin** ({inspect_vec[5]:.2f})
            """)
        with c_f2:
            if st.session_state.get("manual_weights_enabled", False):
                w_dict = st.session_state.manual_weights
                mw = np.array([
                    w_dict.get("rating", 1.5),
                    w_dict.get("price", -1.0),
                    w_dict.get("stock", 0.5),
                    w_dict.get("premium", 1.0),
                    w_dict.get("budget_dist", 1.2),
                    w_dict.get("rating_dist", 0.8)
                ], dtype=np.float32)
                raw_val = np.dot(inspect_vec[:6], mw)
                st.metric("Linear Sum (z)", f"{raw_val:.4f}")
                st.metric("Sigmoid Activation σ(z)", f"{1.0/(1.0+np.exp(-raw_val))*100:.1f}% Match")
            else:
                X_tensor = torch.tensor(inspect_vec, dtype=torch.float32).unsqueeze(0)
                h1 = torch.relu(nn.LayerNorm(64)(_model.net[0](X_tensor)))
                h2 = torch.relu(_model.net[4](h1))
                out_val = _model.net[6](h2).item()
                st.metric("Hidden Activation L2 (Mean)", f"{h2.mean().item():.4f}")
                st.metric("Neural Match Output", f"{out_val*100:.1f}% Match")

# ── 8.3 TAB: CART & WISHLIST CHECKOUTS ────────────────────────────────────────
with t_commerce:
    if st.session_state.get("order_placed", False):
        st.markdown("### 📦 Concierge Fulfillment Dispatch Tracker")
        
        st.markdown("""
        <div style="background: rgba(255,255,255,0.02); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px; margin-bottom: 25px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div style="text-align: center; flex: 1; min-width: 120px;">
                    <h4 style="margin:0; color:#34d399;">✓ PHASE 1</h4>
                    <p style="margin:5px 0 0 0; color:#94a3b8; font-size:0.8rem;">Vault Allocation</p>
                </div>
                <div style="width: 40px; height: 2px; background: #34d399; margin: 0 10px;"></div>
                <div style="text-align: center; flex: 1; min-width: 120px;">
                    <h4 style="margin:0; color:#34d399;">✓ PHASE 2</h4>
                    <p style="margin:5px 0 0 0; color:#94a3b8; font-size:0.8rem;">Bespoke Care Seal</p>
                </div>
                <div style="width: 40px; height: 2px; background: #34d399; margin: 0 10px;"></div>
                <div style="text-align: center; flex: 1; min-width: 120px;">
                    <h4 style="margin:0; color:var(--secondary-glow); animation: pulse 1.5s infinite;">⚡ PHASE 3</h4>
                    <p style="margin:5px 0 0 0; color:#e2e8f0; font-size:0.8rem; font-weight:600;">Armored Courier Transit</p>
                </div>
                <div style="width: 40px; height: 2px; background: rgba(255,255,255,0.1); margin: 0 10px;"></div>
                <div style="text-align: center; flex: 1; min-width: 120px;">
                    <h4 style="margin:0; color:rgba(255,255,255,0.2);">PHASE 4</h4>
                    <p style="margin:5px 0 0 0; color:#94a3b8; font-size:0.8rem;">Client Delivery</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        ord_det = st.session_state.order_details
        items_html = "".join([f"""
            <div style="display:flex; justify-content:space-between; margin-bottom:8px; font-size:0.9rem;">
                <span>{html.escape(item['name'])} (x{item['qty']})</span>
                <span>₹{int(item['price']*item['qty']):,}</span>
            </div>
        """ for item in ord_det["items"]])
        
        st.markdown(f"""
        <div style="background: rgba(10,8,22,0.85); border: 2px solid #fbbf24; border-radius: 16px; padding: 30px; box-shadow: 0 10px 40px rgba(251,191,36,0.1); max-width: 600px; margin: 0 auto; font-family:'Outfit', sans-serif;">
            <div style="text-align:center; margin-bottom: 25px;">
                <h2 style="color:#fbbf24; margin:0; letter-spacing:0.1em;">AURORA CONCIERGE</h2>
                <p style="color:#94a3b8; font-size:0.75rem; letter-spacing:0.2em; text-transform:uppercase; margin:5px 0 0 0;">Certificate of Bespoke Authenticity</p>
            </div>
            <div style="border-top:1px dashed rgba(255,255,255,0.1); border-bottom:1px dashed rgba(255,255,255,0.1); padding: 15px 0; margin-bottom:20px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:10px; font-size:0.85rem; color:#94a3b8;">
                    <span>CERTIFICATE REF</span>
                    <span style="color:#e2e8f0; font-weight:600;">{ord_det['order_id']}</span>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:0.85rem; color:#94a3b8;">
                    <span>TIMESTAMP</span>
                    <span style="color:#e2e8f0; font-weight:600;">{ord_det['timestamp']}</span>
                </div>
            </div>
            <div style="margin-bottom:25px;">
                <p style="font-size:0.75rem; color:#fbbf24; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:12px; font-weight:600;">Allocated Purchases</p>
                {items_html}
            </div>
            <div style="border-top:1px solid rgba(255,255,255,0.1); padding-top:15px; display:flex; justify-content:space-between; margin-bottom:30px;">
                <span style="font-weight:700; font-size:1.1rem; color:#e2e8f0;">TOTAL ESCROW</span>
                <span style="font-weight:700; font-size:1.1rem; color:#34d399;">₹{int(ord_det['total']):,}</span>
            </div>
            <div style="text-align:center; font-size:0.75rem; color:#94a3b8; line-height:1.5;">
                <p>This document verifies that the listed products have been authenticated and securely logged into the client registry. All packaging meets luxury-grade conservation protocol.</p>
                <p style="margin-top:15px; color:#fbbf24; font-weight:600; letter-spacing:0.05em;">AURORA SECURE VERIFIED</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        if st.button("Return to Curation Panel", use_container_width=True):
            st.session_state.order_placed = False
            st.session_state.order_details = None
            st.rerun()
    else:
        c_comm1, c_comm2 = st.columns(2)
        with c_comm1:
            st.markdown("<h3 style='color: #ec4899;'>🛒 Shopping Cart</h3>", unsafe_allow_html=True)
            cart = st.session_state.shopping_cart
            if not cart:
                st.info("Shopping Cart is empty. Click 'Add to Cart' inside recommended items.")
            else:
                total_sum = 0.0
                for p_id, item in list(cart.items()):
                    sub_total = item["price"] * item["qty"]
                    total_sum += sub_total

                    c_cart1, c_cart2, c_cart3 = st.columns([4, 2, 2])
                    c_cart1.markdown(f"**{html.escape(item['name'])}**  \n₹{int(item['price']):,} each")
                    
                    new_qty = c_cart2.number_input("Qty", min_value=0, max_value=100, value=int(item["qty"]), key=f"q_{p_id}")
                    if new_qty == 0:
                        del cart[p_id]
                        st.rerun()
                    elif new_qty != item["qty"]:
                        cart[p_id]["qty"] = new_qty
                        st.rerun()

                    c_cart3.markdown(f"  \n**₹{int(sub_total):,}**")

                st.markdown("---")
                st.markdown(f"#### Total Cart Amount: <span style='color: #34d399;'>₹{int(total_sum):,}</span>", unsafe_allow_html=True)

                if st.button("Place Concierge Order", use_container_width=True, type="primary"):
                    st.balloons()
                    
                    # Store order metadata
                    order_id = f"AUR-{random.randint(1000,9999)}-{int(time.time()) % 100000}"
                    cart_items = list(cart.values())
                    st.session_state.order_details = {
                        "order_id": order_id,
                        "items": cart_items,
                        "total": total_sum,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    }
                    st.session_state.order_placed = True
                    st.session_state.shopping_cart.clear()
                    st.rerun()

        with c_comm2:
            st.markdown("<h3 style='color: #f43f5e;'>❤️ Personal Wishlist</h3>", unsafe_allow_html=True)
            wishlist = st.session_state.wishlist
            if not wishlist:
                st.info("Wishlist is empty. Tap 'Save Wishlist' on items to remember them.")
            else:
                wish_ids = list(wishlist)
                sql = f"SELECT id, name, price, rating, category FROM products WHERE id IN ({','.join(['?']*len(wish_ids))})"
                with sqlite3.connect(DB_FILE) as conn:
                    wish_df = pd.read_sql_query(sql, conn, params=wish_ids)

                for _, w_row in wish_df.iterrows():
                    c_wish1, c_wish2 = st.columns([5, 3])
                    c_wish1.markdown(f"**{html.escape(w_row['name'])}**  \n₹{int(w_row['price']):,} &middot; ★ {w_row['rating']:.1f}")

                    if c_wish2.button("Move to Cart", key=f"wish_mv_{w_row['id']}", use_container_width=True):
                        cart = st.session_state.shopping_cart
                        p_id_str = str(w_row["id"])
                        if p_id_str in cart:
                            cart[p_id_str]["qty"] += 1
                        else:
                            cart[p_id_str] = {"name": w_row["name"], "price": float(w_row["price"]), "qty": 1}
                        st.session_state.wishlist.remove(w_row["id"])
                        st.toast("Moved item into shopping cart.", icon="🛒")
                        st.rerun()

                    st.markdown("<hr style='border-color:rgba(255,255,255,0.05); margin:8px 0;'>", unsafe_allow_html=True)

# ── 8.4 TAB: COMPLETE DATABASE EXPLORER ───────────────────────────────────────
with t_db:
    st.markdown("<h3 style='margin-top: 10px;'>Inventory database manager</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8;'>Search, filter and page across the entire set of 2,000 product rows in real-time.</p>", unsafe_allow_html=True)

    col_filt1, col_filt2, col_filt3 = st.columns(3)
    db_cat = col_filt1.selectbox("Filter Category", ["All Categories"] + [c.replace("_", " ").title() for c in CATEGORIES])
    db_search = col_filt2.text_input("Find by Product Name", value="")
    db_sort = col_filt3.selectbox("Sort Ordering", ["Price (Low to High)", "Price (High to Low)", "Ratings (Highest first)"])

    sql_db = "SELECT id, name, category, price, rating, stock FROM products WHERE 1=1"
    params_db = []

    if db_cat != "All Categories":
        sql_db += " AND category = ?"
        params_db.append(db_cat.lower().replace(" ", "_"))

    if db_search.strip():
        sql_db += " AND name LIKE ?"
        params_db.append(f"%{db_search.strip()}%")

    if db_sort == "Price (Low to High)":
        sql_db += " ORDER BY price ASC"
    elif db_sort == "Price (High to Low)":
        sql_db += " ORDER BY price DESC"
    else:
        sql_db += " ORDER BY rating DESC, price ASC"

    with sqlite3.connect(DB_FILE) as conn:
        all_inventory = pd.read_sql_query(sql_db, conn, params=params_db)

    if all_inventory.empty:
        st.info("No items in inventory matched the filter constraints.")
    else:
        page_size = 15
        total_rows = len(all_inventory)
        total_pages = int(np.ceil(total_rows / page_size))

        c_pg1, c_pg2 = st.columns([1, 4])
        target_page = c_pg1.number_input("Browse Page", min_value=1, max_value=total_pages, value=1, step=1)
        c_pg2.markdown(f"  \nShowing **{(target_page-1)*page_size + 1}** to **{min(target_page*page_size, total_rows)}** items out of **{total_rows}** rows total.")

        start_slice = (target_page - 1) * page_size
        sliced_df = all_inventory.iloc[start_slice : start_slice + page_size].reset_index(drop=True)

        st.dataframe(
            sliced_df,
            column_config={
                "id": "Product ID",
                "name": "Product Name",
                "category": "Category Class",
                "price": st.column_config.NumberColumn("Unit Price (₹)", format="₹%d"),
                "rating": st.column_config.NumberColumn("Product Stars", format="★ %.1f"),
                "stock": "Units Stock"
            },
            hide_index=True,
            use_container_width=True
        )

# ── 8.5 TAB: DISMISSED EXCLUSIONS ─────────────────────────────────────────────
with t_exclusions:
    st.markdown("<h3 style='margin-top: 10px;'>Excluded items list</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8;'>Browse items you have dismissed from recommendation arrays during matching sessions.</p>", unsafe_allow_html=True)

    blacklist = st.session_state.blacklist
    if not blacklist:
        st.info("The exclusions registry is currently empty.")
    else:
        bl_ids = list(blacklist)
        sql = f"SELECT id, name, price, rating, category FROM products WHERE id IN ({','.join(['?']*len(bl_ids))})"
        with sqlite3.connect(DB_FILE) as conn:
            blacklisted_df = pd.read_sql_query(sql, conn, params=bl_ids)

        for _, bl_row in blacklisted_df.iterrows():
            c_bl1, c_bl2 = st.columns([5, 3])
            c_bl1.markdown(f"**{html.escape(bl_row['name'])}**  \n₹{int(bl_row['price']):,} &middot; ★ {bl_row['rating']:.1f}")

            if c_bl2.button("Restore Selection", key=f"bl_rest_{bl_row['id']}", use_container_width=True):
                st.session_state.blacklist.remove(bl_row["id"])
                st.toast(f"Restored {bl_row['name']} to curated streams.", icon="🔄")
                st.rerun()

            st.markdown("<hr style='border-color:rgba(255,255,255,0.05); margin:8px 0;'>", unsafe_allow_html=True)
