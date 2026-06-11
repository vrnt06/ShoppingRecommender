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
user_cat = st.sidebar.selectbox("Collection Portfolio",
