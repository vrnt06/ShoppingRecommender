import os
import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim

# Apply a premium style and layout config
st.set_page_config(page_title="AURORA // Premium AI Concierge", page_icon="⚡", layout="wide")

# Custom CSS injection for a minimalist Nordic styling theme
st.markdown("""
    <style>
    /* Global Background and typography adjustments */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }
    /* Clean Product Container styling Cards */
    .product-card {
        background: linear-gradient(145deg, #161b22, #0f141c);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .product-card:hover {
        border-color: #58a6ff;
    }
    /* Brand Accent headers */
    .brand-title {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #f0f6fc;
    }
    /* Match Pill Style */
    .match-pill {
        background-color: rgba(56, 139, 253, 0.15);
        color: #58a6ff;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        border: 1px solid rgba(56, 139, 253, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

DB_FILE = "inventory.db"
CATEGORIES = ["electronics", "footwear", "clothing", "beauty", "home_decor", "fitness", "books", "automotive", "toys", "groceries"]
FEATURE_DIM = 12
MAX_GLOBAL_PRICE = 250000.0

if not os.path.exists(DB_FILE):
    st.error("🚨 Core inventory missing. Please run seed_db.py to generate your database items.")
    st.stop()

# ==========================================
# 1. CORE SEARCH & EMBEDDING ENGINES
# ==========================================
def build_vectors_from_df(df_subset):
    vectors = []
    for _, row in df_subset.iterrows():
        rating_val = float(row["rating"])
        price_norm = float(row["price"]) / MAX_GLOBAL_PRICE
        cat_array = [0.0] * len(CATEGORIES)
        if row["category"] in CATEGORIES:
            cat_idx = CATEGORIES.index(row["category"])
            cat_array[cat_idx] = 1.0
        vectors.append([rating_val, price_norm] + cat_array)
    return np.array(vectors, dtype=np.float32)

def parse_user_input(category_choice, max_budget, preferred_rating):
    price_norm = max_budget / MAX_GLOBAL_PRICE
    cat_array = [0.0] * len(CATEGORIES)
    if category_choice in CATEGORIES:
        cat_idx = CATEGORIES.index(category_choice)
        cat_array[cat_idx] = 1.0
    return np.array([preferred_rating, price_norm] + cat_array, dtype=np.float32)

def query_candidates_db(user_vector, category_choice, max_budget, blacklist, search_query=""):
    conn = sqlite3.connect(DB_FILE)
    query = "SELECT id, name, category, price, rating FROM products WHERE price <= ? AND category = ?"
    params = [max_budget, category_choice]
    
    if search_query.strip():
        query += " AND name LIKE ?"
        params.append(f"%{search_query.strip()}%")
        
    if blacklist:
        placeholders = ",".join("?" for _ in blacklist)
        query += f" AND id NOT IN ({placeholders})"
        params.extend(list(blacklist))
        
    df_candidates = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    if df_candidates.empty:
        return df_candidates, np.empty((0, FEATURE_DIM), dtype=np.float32)
    
    candidate_vectors = build_vectors_from_df(df_candidates)
    sims = cosine_similarity([user_vector], candidate_vectors)[0]
    
    actual_k = min(3, len(df_candidates))
    top_indices = np.argsort(sims)[-actual_k:]
    
    return df_candidates.iloc[top_indices].copy(), candidate_vectors[top_indices]

# ==========================================
# 2. NEURAL NETWORK LEARNING PIPELINE
# ==========================================
class DeepRanker(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

if "neural_agent" not in st.session_state:
    st.session_state.neural_agent = DeepRanker(FEATURE_DIM)

agent_nn = st.session_state.neural_agent

def instant_backprop_step(feature_vector, target_label):
    X = torch.tensor(np.array([feature_vector])).float()
    y = torch.tensor(np.array([[target_label]])).float()
    
    optimizer = optim.Adam(agent_nn.parameters(), lr=0.05)
    loss_fn = nn.BCELoss()
    
    agent_nn.train()
    for _ in range(12):
        predictions = agent_nn(X)
        loss = loss_fn(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# --- INSTANT PERSISTENT REVIEWS CONTROLLER ---
if "persistent_reviews" not in st.session_state:
    st.session_state.persistent_reviews = {}

def get_product_reviews(product_name):
    if product_name not in st.session_state.persistent_reviews:
        st.session_state.persistent_reviews[product_name] = [
            {"user": "Alex M.", "stars": 5, "comment": "Excellent craftsmanship. Exceeded expectations."},
            {"user": "S. Taylor", "stars": 4, "comment": "Solid everyday design, sleek minimalist appearance."}
        ]
    return st.session_state.persistent_reviews[product_name]

# ==========================================
# 3. LUXURY NORDIC CUSTOMER INTERFACE
# ==========================================
if "blacklist" not in st.session_state:
    st.session_state.blacklist = set()

# Left Premium Sidebar Control Deck
st.sidebar.markdown("<h2 class='brand-title'>AURORA // DECK</h2>", unsafe_allow_html=True)
user_cat = st.sidebar.selectbox("Collection Portfolio", [c.replace("_", " ").title() for c in CATEGORIES], index=0)
selected_category_key = user_cat.lower().replace(" ", "_")

user_budget = st.sidebar.slider("Investment Range Ceiling (₹)", min_value=100, max_value=250000, value=170000, step=1000)
user_rating = st.sidebar.slider("Minimum Grade Quality", min_value=1.0, max_value=5.0, value=4.0, step=0.1)

st.sidebar.markdown("---")
if st.sidebar.button("Reset Style History Profile", use_container_width=True):
    st.session_state.blacklist.clear()
    st.toast("Profile memory refreshed.")
    st.rerun()

# Workspace Header Layout
st.markdown("<h1 class='brand-title'>AURORA</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #8b949e; font-size: 1.1rem;'>Curated luxury items matched instantly to your aesthetic signature.</p>", unsafe_allow_html=True)

search_input = st.text_input("Filter portfolio selection by specific keywords...", value="", placeholder="Search model variants, brands, series...")

# Core Query Operations Execution Block
u_vector = parse_user_input(selected_category_key, user_budget, user_rating)
candidates, vectors = query_candidates_db(u_vector, selected_category_key, user_budget, st.session_state.blacklist, search_query=search_input)

if not candidates.empty:
    agent_nn.eval()
    with torch.no_grad():
        scores = agent_nn(torch.tensor(vectors).float()).numpy().flatten()
    
    candidates["score"] = scores
    ranked_output = candidates.sort_values(by="score", ascending=False)
    
    st.markdown("<p style='color: #8b949e; margin-top:20px; text-transform: uppercase; font-size: 0.8rem; letter-spacing: 0.1em;'>Suggested Curations</p>", unsafe_allow_html=True)
    
    # Generate the curated product listings
    for idx, (_, row) in enumerate(ranked_output.iterrows()):
        item_vector = vectors[idx]
        match_percentage = int(row['score'] * 100)
        
        # Open card container block
        st.markdown(f"""
            <div class='product-card'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <h3 style='margin: 0; color: #f0f6fc; font-weight: 500;'>{row['name']}</h3>
                    <div class='match-pill'>✦ {match_percentage}% MATCH</div>
                </div>
                <p style='color: #8b949e; margin: 8px 0 16px 0; font-size: 0.95rem;'>
                    Collection Category: <span style='color: #c9d1d9;'>{row['category'].upper()}</span> &nbsp;|&nbsp; 
                    Market Value: <span style='color: #58a6ff;'>₹{int(row['price']):,}</span> &nbsp;|&nbsp; 
                    Evaluation: <span style='color: #ffd33d;'>★ {row['rating']} / 5.0</span>
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Insert interaction widgets cleanly underneath the custom formatted HTML container block
        col_rev, col_act = st.columns([2.5, 1])
        
        with col_rev:
            reviews_list = get_product_reviews(row['name'])
            with st.expander(f"Verified Buyer Reviews ({len(reviews_list)})"):
                for r in reviews_list:
                    st.markdown(f"**{r['user']}** &nbsp;<span style='color:#ffd33d;'>{'★' * r['stars']}{'☆' * (5-r['stars'])}</span>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color:#8b949e; font-style: italic; font-size:0.9rem;'>\"{r['comment']}\"</p>", unsafe_allow_html=True)
                
                # Active inline review submission form interface
                with st.form(key=f"rev_nordic_{row['id']}", clear_on_submit=True):
                    inner_col1, inner_col2 = st.columns([3, 1])
                    u_name = inner_col1.text_input("Name", value="Anonymous", key=f"un_{row['id']}")
                    u_stars = inner_col2.slider("Stars", 1, 5, 5, key=f"us_{row['id']}")
                    u_comment = st.text_area("Your Review Feedback", placeholder="Share your assessment regarding performance or build quality...", key=f"uc_{row['id']}")
                    
                    if st.form_submit_button("Publish Anonymous Review"):
                        if u_comment.strip():
                            st.session_state.persistent_reviews[row['name']].insert(0, {
                                "user": u_name,
                                "stars": u_stars,
                                "comment": u_comment.strip()
                            })
                            st.toast("Review recorded.")
                            st.rerun()
                            
        with col_act:
            btn_col1, btn_col2 = st.columns(2)
            # Automatic reaction hooks update the agent vector spaces instantly without requiring submit boxes
            if btn_col1.button("✨ Keep", key=f"like_{row['id']}", use_container_width=True):
                instant_backprop_step(item_vector, 1.0)
                st.toast(f"Preference logged for {row['name']}")
                st.rerun()
                
            if btn_col2.button("Dismiss", key=f"dismiss_{row['id']}", use_container_width=True):
                instant_backprop_step(item_vector, 0.0)
                st.session_state.blacklist.add(row['id'])
                st.toast(f"Removed {row['name']}")
                st.rerun()
                
        st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)
else:
    st.markdown("---")
    st.warning("No items fit your specific filters. Adjust your category criteria or expand your investment range sliders on the left menu.")
