import os
import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim

st.set_page_config(page_title="Enterprise DB AI Agent", page_icon="💾", layout="wide")

DB_FILE = "inventory.db"
CATEGORIES = ["electronics", "footwear", "clothing", "beauty", "home_decor", "fitness", "books", "automotive", "toys", "groceries"]
FEATURE_DIM = 12  # Structure: [rating, price_norm, 10 category one-hot flags]
MAX_GLOBAL_PRICE = 150000.0  # Normalized pricing ceiling anchor

# ==========================================
# 1. DATABASE INITIALIZATION & SEEDING
# ==========================================
def init_db():
    """Establishes connections and initializes relational schema constraints."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Create production schema
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            rating REAL NOT NULL
        )
    """)
    
    # Check if table contains data; seed with real sample records if empty
    cursor.execute("SELECT COUNT(*) FROM products")
    if cursor.fetchone()[0] == 0:
        initial_inventory = [
            # Electronics
            ("iPhone 15 Pro", "electronics", 129900, 4.8),
            ("MacBook Air M3", "electronics", 114900, 4.7),
            ("Sony WH-1000XM5 Headphones", "electronics", 29990, 4.6),
            ("iPad Air", "electronics", 59900, 4.5),
            ("Dell UltraSharp 27 Monitor", "electronics", 34500, 4.4),
            ("Logitech MX Master 3S Mouse", "electronics", 9495, 4.7),
            # Footwear
            ("Nike Air Max Alpha", "footwear", 7995, 4.3),
            ("Adidas Ultraboost Light", "footwear", 17999, 4.6),
            ("Puma Velocity Nitro", "footwear", 11999, 4.4),
            ("Asics Gel-Kayano 30", "footwear", 15999, 4.8),
            # Clothing
            ("Levi's 511 Slim Fit Jeans", "clothing", 4199, 4.2),
            ("Patagonia Torrentshell Jacket", "clothing", 14999, 4.7),
            ("North Face Nuptse Down Jacket", "clothing", 27999, 4.8),
            # Books
            ("Designing Data-Intensive Applications", "books", 1850, 4.9),
            ("Deep Learning by Goodfellow", "books", 4200, 4.7),
            ("Atomic Habits", "books", 550, 4.8)
        ]
        
        # Insert seed data block
        cursor.executemany("""
            INSERT INTO products (name, category, price, rating) 
            VALUES (?, ?, ?, ?)
        """, initial_inventory)
        conn.commit()
        
    conn.close()

# Run database setup verification
init_db()

# ==========================================
# 2. MATRIX EMBEDDING & VECTOR ENGINES
# ==========================================
def build_vectors_from_df(df_subset):
    """Transforms raw database frames into structured floating-point vector tensors."""
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
    """Stage 1: SQL Execution Layer (Handles hard budget constraints at database layer)"""
    conn = sqlite3.connect(DB_FILE)
    
    # Base SQL structural assignment
    query = "SELECT id, name, category, price, rating FROM products WHERE price <= ?"
    params = [max_budget]
    
    # Append textual keyword filter conditions dynamically
    if search_query.strip():
        query += " AND name LIKE ?"
        params.append(f"%{search_query.strip()}%")
    else:
        query += " AND category = ?"
        params.append(category_choice)
        
    # Append dynamic session exclusions
    if blacklist:
        placeholders = ",".join("?" for _ in blacklist)
        query += f" AND id NOT IN ({placeholders})"
        params.extend(list(blacklist))
        
    df_candidates = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    if df_candidates.empty:
        return df_candidates, np.empty((0, FEATURE_DIM), dtype=np.float32)
    
    # Process relative mathematical distance vectors
    candidate_vectors = build_vectors_from_df(df_candidates)
    sims = cosine_similarity([user_vector], candidate_vectors)[0]
    
    # Sort closest contextual vector arrays
    actual_k = min(3, len(df_candidates))
    top_indices = np.argsort(sims)[-actual_k:]
    
    return df_candidates.iloc[top_indices].copy(), candidate_vectors[top_indices]

# ==========================================
# 3. PYTORCH DEEP RANKING PIPELINE
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
# 4. STREAMLIT ENTERPRISE INTERFACE
# ==========================================
st.title("💾 Database-Driven AI Recommendation Agent")
st.caption("Production Pipeline Connected to Local SQLite Engine with In-Memory Neural Topologies.")

if "blacklist" not in st.session_state:
    st.session_state.blacklist = set()

if "current_recommendations" not in st.session_state:
    st.session_state.current_recommendations = None

# Search input engine trigger
search_input = st.text_input("🔍 Search Inventory Database (e.g., 'iPhone', 'Nike', 'MacBook')", value="")

st.sidebar.header("🎯 Context Filters")
user_cat = st.sidebar.selectbox("Fallback Category", [c.replace("_", " ").title() for c in CATEGORIES])
selected_category_key = user_cat.lower().replace(" ", "_")

user_budget = st.sidebar.slider("Maximum Budget Cap (₹)", min_value=500, max_value=150000, value=75000, step=500)
user_rating = st.sidebar.slider("Minimum Quality Metric", min_value=1.0, max_value=5.0, value=4.0, step=0.1)

if st.sidebar.button("🧹 Clear Dislike Blacklist"):
    st.session_state.blacklist.clear()
    st.session_state.current_recommendations = None
    st.rerun()

# Processing Loop Block execution
if st.button("🧠 Search & Compute Next Best Action", type="primary", use_container_width=True):
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

# Output rendering form engine
if st.session_state.current_recommendations is not None:
    if st.session_state.current_recommendations == "EMPTY":
        st.error("❌ No matching rows discovered in SQLite table. Try widening the scope parameters or budget levels.")
    else:
        ranked_df, vectors_used = st.session_state.current_recommendations
        st.subheader("💡 Database Query Results & Neural Ranking Scores")
        
        with st.form("feedback_form"):
            feedback_dict = {}
            
            for i, (idx, row) in enumerate(ranked_df.iterrows()):
                col_item, col_feed = st.columns([3, 1])
                with col_item:
                    st.info(f"**{row['name']}** [{row['category'].upper()}] \n💰 Price: ₹{row['price']:,} | ⭐ Rating: {row['rating']} | 🕸️ Current Layer Score: `{round(row['score'], 4)}`")
                with col_feed:
                    feedback_dict[idx] = st.radio("Feedback", ["Select Action", "👍 Like / Buy", "👎 Dislike / Ignore"], key=f"feed_{row['id']}")
                    
            submitted = st.form_submit_button("📥 Process Feedback & Train Neural Layers", use_container_width=True)
            
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
                    with st.spinner("Executing neural learning backpropagation..."):
                        dynamic_train_step(training_features, training_labels)
                    st.success("🤖 Optimization Complete! System weights adjusted based on database evaluations.")
                    st.session_state.current_recommendations = None
                    st.rerun()
                else:
                    st.warning("Provide evaluation metrics to activate backpropagation optimization cycles.")
else:
    st.write("### 👈 Query your persistent SQLite tables by filling out search inputs.")
