import os
import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim

st.set_page_config(page_title="AI Personal Concierge", page_icon="🛍️", layout="wide")

DB_FILE = "inventory.db"
CATEGORIES = ["electronics", "footwear", "clothing", "beauty", "home_decor", "fitness", "books", "automotive", "toys", "groceries"]
FEATURE_DIM = 12
MAX_GLOBAL_PRICE = 250000.0

if not os.path.exists(DB_FILE):
    st.error("🚨 Missing 'inventory.db' file! Please run your database creator script first.")
    st.stop()

# ==========================================
# 1. MATHEMATICAL DATA & LOOKUP PIPELINES
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
# 2. SEAMLESS AI OPTIMIZATION BACKEND
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
    """Fires instantly behind the scenes on button click—no submission needed."""
    X = torch.tensor(np.array([feature_vector])).float()
    y = torch.tensor(np.array([[target_label]])).float()
    
    optimizer = optim.Adam(agent_nn.parameters(), lr=0.05)
    loss_fn = nn.BCELoss()
    
    agent_nn.train()
    for _ in range(10):
        predictions = agent_nn(X)
        loss = loss_fn(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# --- REVIEWS & FEEDBACK STORAGE SYSTEM ---
if "persistent_reviews" not in st.session_state:
    st.session_state.persistent_reviews = {}

def get_product_reviews(product_name):
    # Generates custom sample responses based on keywords or pulls up written submissions
    if product_name not in st.session_state.persistent_reviews:
        brand = product_name.split()[0]
        if brand in ["Apple", "Samsung", "Sony"]:
            st.session_state.persistent_reviews[product_name] = [
                {"user": "Arjun R.", "stars": 5, "comment": "Absolutely brilliant choice. Premium build and exceptional performance."},
                {"user": "Priya S.", "stars": 4, "comment": "Incredibly smooth user interface, but a bit pricey."}
            ]
        elif brand in ["Nike", "Adidas", "Puma"]:
            st.session_state.persistent_reviews[product_name] = [
                {"user": "Vikram K.", "stars": 5, "comment": "Super lightweight and highly comfortable for running long tracks."},
                {"user": "Ananya D.", "stars": 3, "comment": "Arch support is fantastic, but fits tighter than standard sizing."}
            ]
        else:
            st.session_state.persistent_reviews[product_name] = [
                {"user": "Verified Shopper", "stars": 5, "comment": "Highly recommended! Exceeded my expectations entirely."},
                {"user": "Rohan M.", "stars": 4, "comment": "Good quality and value. Shipped quickly and arrived safely."}
            ]
    return st.session_state.persistent_reviews[product_name]

# ==========================================
# 3. INTERACTIVE CUSTOMER FACE LAYOUT
# ==========================================
if "blacklist" not in st.session_state:
    st.session_state.blacklist = set()

# Sidebar Search Customization Context
st.sidebar.markdown("### 🔍 Personalize Your Stylist")
user_cat = st.sidebar.selectbox("Category", [c.replace("_", " ").title() for c in CATEGORIES], index=0)
selected_category_key = user_cat.lower().replace(" ", "_")

user_budget = st.sidebar.slider("Max Budget Target (₹)", min_value=100, max_value=250000, value=160000, step=1000)
user_rating = st.sidebar.slider("Minimum Customer Rating", min_value=1.0, max_value=5.0, value=4.0, step=0.1)

if st.sidebar.button("🧹 Reset Personalization Model", use_container_width=True):
    st.session_state.blacklist.clear()
    if "current_recommendations" in st.session_state:
        st.session_state.current_recommendations = None
    st.rerun()

# Clean Header Design Structure
st.title("🛍️ Your Personalized AI Stylist")
st.write("Browse through our live collection. Your assistant fine-tunes your feed automatically based on your preferences.")

search_input = st.text_input(f"What specifically are you shopping for in {user_cat}? (Leave blank to see everything)", value="", placeholder="Type brands, keywords, models...")

# Reactive Computation Layer
u_vector = parse_user_input(selected_category_key, user_budget, user_rating)
candidates, vectors = query_candidates_db(u_vector, selected_category_key, user_budget, st.session_state.blacklist, search_query=search_input)

if not candidates.empty:
    agent_nn.eval()
    with torch.no_grad():
        scores = agent_nn(torch.tensor(vectors).float()).numpy().flatten()
    
    candidates["score"] = scores
    ranked_output = candidates.sort_values(by="score", ascending=False)
    
    st.markdown("---")
    st.subheader(f"✨ Curated Top Matches for You in {user_cat}")
    
    # Render individual beautiful item catalog cards dynamically
    for idx, (_, row) in enumerate(ranked_output.iterrows()):
        item_vector = vectors[idx]
        match_percentage = int(row['score'] * 100)
        
        # UI Product Block Card styling container wrapper
        with st.container():
            col_details, col_actions = st.columns([2.5, 1], gap="medium")
            
            with col_details:
                st.markdown(f"### {row['name']}")
                
                # Align key statistics neatly
                m1, m2, m3 = st.columns(3)
                m1.metric("Price", f"₹{int(row['price']):,}")
                m2.metric("Avg Rating", f"⭐ {row['rating']}/5")
                m3.metric("Tailored Match", f"🔥 {match_percentage}%")
                
                # --- INTERACTIVE REVIEW EXTENSION WINDOW ---
                reviews = get_product_reviews(row['name'])
                with st.expander(f"💬 Customer Reviews & Feedback ({len(reviews)})"):
                    for r in reviews:
                        st.markdown(f"**{r['user']}** {'★' * r['stars']}{'☆' * (5-r['stars'])}")
                        st.caption(f'"{r["comment"]}"')
                        st.markdown("<p style='margin:2px;'></p>", unsafe_allow_html=True)
                    
                    # Form enabling users to write review items
                    with st.form(key=f"write_review_{row['id']}", clear_on_submit=True):
                        st.markdown("**Write a Product Review**")
                        rev_name = st.text_input("Your Name", value="Guest User", key=f"rn_{row['id']}")
                        rev_stars = st.slider("Rating Stars", 1, 5, 5, key=f"rs_{row['id']}")
                        rev_text = st.text_area("Your Review Thoughts", placeholder="Share your experience with this item...", key=f"rt_{row['id']}")
                        
                        if st.form_submit_button("Post Anonymous Review"):
                            if rev_text.strip():
                                st.session_state.persistent_reviews[row['name']].insert(0, {
                                    "user": rev_name,
                                    "stars": rev_stars,
                                    "comment": rev_text.strip()
                                })
                                st.toast("✅ Review posted successfully!")
                                st.rerun()
            
            with col_actions:
                st.write("<p style='margin-top:25px;'></p>", unsafe_allow_html=True)
                
                # INTERACTION FUNCTIONALITIES: Automatically fires training backprop instantly when pushed
                if st.button("👍 Love It", key=f"like_{row['id']}", use_container_width=True):
                    instant_backprop_step(item_vector, 1.0)
                    st.toast(f"Loved {row['name']}! Refining your feed...")
                    st.rerun()
                    
                if st.button("👎 Hide From Feed", key=f"hide_{row['id']}", use_container_width=True):
                    instant_backprop_step(item_vector, 0.0)
                    st.session_state.blacklist.add(row['id'])
                    st.toast(f"Removed {row['name']} from your recommendations.")
                    st.rerun()
                    
        st.markdown("<hr style='border:1px solid #f0f2f6; margin-top:20px; margin-bottom:20px;'>", unsafe_allow_html=True)
else:
    st.markdown("---")
    st.error(f" We couldn't find any products in '{user_cat}' matching your filter settings. Try raising your budget or clearing your search text query above.")
