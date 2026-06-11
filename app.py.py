import os
import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim

st.set_page_config(page_title="Production Relational AI Agent", page_icon="🤖", layout="wide")

DB_FILE = "inventory.db"
CATEGORIES = ["electronics", "footwear", "clothing", "beauty", "home_decor", "fitness", "books", "automotive", "toys", "groceries"]
FEATURE_DIM = 12
MAX_GLOBAL_PRICE = 180000.0  # Synced maximum pricing anchor across database ranges

# Check for database presence before executing front-end rendering
if not os.path.exists(DB_FILE):
    st.error("🚨 Missing 'inventory.db' file! Please run 'python seed_db.py' first to build and populate your 2,000 product rows.")
    st.stop()

# ==========================================
# 1. STRUCTURAL DATA & EMBEDDING PIPELINES
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
    """Performs Stage 1 relational lookup at database core layer."""
    conn = sqlite3.connect(DB_FILE)
    query = "SELECT id, name, category, price, rating FROM products WHERE price <= ?"
    params = [max_budget]
    
    if search_query.strip():
        query += " AND name LIKE ?"
        params.append(f"%{search_query.strip()}%")
    else:
        query += " AND category = ?"
        params.append(category_choice)
        
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
    
    # Rank top 3 items within filtering sub-selections
    actual_k = min(3, len(df_candidates))
    top_indices = np.argsort(sims)[-actual_k:]
    
    return df_candidates.iloc[top_indices].copy(), candidate_vectors[top_indices]

# ==========================================
# 2. PYTORCH CONTINUOUS TRAINING ENGINE
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

def dynamic_train_step(features, labels):
    X = torch.tensor(np.array(features)).float()
    y = torch.tensor(np.array(labels)).float().unsqueeze(1)
    
    optimizer = optim.Adam(agent_nn.parameters(), lr=0.05)
    loss_fn = nn.BCELoss()
    
    agent_nn.train()
    for _ in range(15):
        predictions = agent_nn(X)
        loss = loss_fn(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# ==========================================
# 3. INTERACTIVE RENDERING CANVAS
# ==========================================
st.title("🤖 Production Deep Ranking Recommendation Agent")
st.caption("Active SQL Backend Processing 2,000 Real Items (200 SKUs Per Category)")

if "blacklist" not in st.session_state:
    st.session_state.blacklist = set()

if "current_recommendations" not in st.session_state:
    st.session_state.current_recommendations = None

search_input = st.text_input("🔍 Search Entire 2,000 Product Database (e.g., 'iPhone', 'Ultraboost', 'LEGO', 'Dark Chocolate')", value="")

st.sidebar.header("🎯 Target Context Weights")
user_cat = st.sidebar.selectbox("Fallback Category", [c.replace("_", " ").title() for c in CATEGORIES])
selected_category_key = user_cat.lower().replace(" ", "_")

user_budget = st.sidebar.slider("Maximum Budget Target (₹)", min_value=100, max_value=250000, value=120000, step=1000)
user_rating = st.sidebar.slider("Minimum Quality Target", min_value=1.0, max_value=5.0, value=4.0, step=0.1)

if st.sidebar.button("🧹 Clear Dislike Blacklist"):
    st.session_state.blacklist.clear()
    st.session_state.current_recommendations = None
    st.rerun()

if st.button("🧠 Execute Matrix Search & Rank", type="primary", use_container_width=True):
    u_vector = parse_user_input(selected_category_key, user_budget, user_rating)
    candidates, vectors = query_candidates_db(u_vector, selected_category_key, user_budget, st.session_state.blacklist, search_query=search_input)
    
    if not candidates.empty:
        agent_nn.eval()
        with torch.no_grad():
            scores = agent_nn(torch.tensor(vectors).float()).numpy().flatten()
        
        candidates["score"] = scores
        ranked_output = candidates.sort_values(by="score", ascending=False)
        st.session_state.current_recommendations = (ranked_output, vectors)
    else:
        st.session_state.current_recommendations = "EMPTY"

if st.session_state.current_recommendations is not None:
    if st.session_state.current_recommendations == "EMPTY":
        st.error("❌ No matching records found. Refine your text keywords or increase pricing budget sliders.")
    else:
        ranked_df, vectors_used = st.session_state.current_recommendations
        st.subheader("💡 Retrieved Database Rows & Current Layer Scoring Values")
        
        with st.form("feedback_form"):
            feedback_dict = {}
            
            for i, (idx, row) in enumerate(ranked_df.iterrows()):
                col_item, col_feed = st.columns([3, 1])
                with col_item:
                    st.info(f"**{row['name']}** [{row['category'].upper()}] \n💰 Price: ₹{row['price']:,} | ⭐ Rating: {row['rating']} | 🕸️ Current Layer Score: `{round(row['score'], 4)}`")
                with col_feed:
                    feedback_dict[idx] = st.radio("Feedback Action", ["Select Action", "👍 Like / Buy", "👎 Dislike / Ignore"], key=f"feed_{row['id']}")
                    
            submitted = st.form_submit_button("📥 Send Choices to PyTorch Neural Gradients", use_container_width=True)
            
            if submitted:
                training_features = []
                training_labels = []
                
                for i, (idx, row) in enumerate(ranked_df.iterrows()):
                    action = feedback_dict[idx]
                    if action != "Select Action":
                        training_features.append(vectors_used[i])
                        training_labels.append(1.0 if action == "👍 Like / Buy" else 0.0)
                        
                        if action == "👎 Dislike / Ignore":
                            st.session_state.blacklist.add(row['id'])
            
                if training_features:
                    with st.spinner("Updating backpropagation neural vectors..."):
                        dynamic_train_step(training_features, training_labels)
                    st.success("🤖 PyTorch network updated! Recommendations optimized based on database selections.")
                    st.session_state.current_recommendations = None
                    st.rerun()
                else:
                    st.warning("Provide implicit input rankings to run structural weight revisions.")
else:
    st.write("### 👈 Apply search parameters or filter limits to query the SQL stack.")
