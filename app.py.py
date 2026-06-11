import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim

st.set_page_config(page_title="AI Agent (Zero-Cache Mode)", page_icon="🤖", layout="wide")

# ==========================================
# 1. LIGHTWEIGHT IN-MEMORY REGISTRY
# ==========================================
CATEGORIES = [
    "electronics", "footwear", "clothing", "beauty", "home_decor",
    "fitness", "books", "automotive", "toys", "groceries"
]

category_archetypes = {
    "electronics": {"prefixes": ["Pro", "Ultra", "Max"], "names": ["Laptop", "Smartphone", "Headphones", "Monitor"], "min_p": 1500, "max_p": 95000},
    "footwear": {"prefixes": ["Air", "Gel", "Zoom"], "names": ["Runners", "Sneakers", "Formals", "Boots"], "min_p": 800, "max_p": 18000},
    "clothing": {"prefixes": ["Urban", "Slim Fit", "Oversized"], "names": ["Hoodie", "Jacket", "T-Shirt", "Chinos"], "min_p": 500, "max_p": 12000},
    "beauty": {"prefixes": ["Hydra", "Glow", "Matte"], "names": ["Serum", "Moisturizer", "Lipstick", "Sunscreen"], "min_p": 250, "max_p": 6000},
    "home_decor": {"prefixes": ["Nordic", "Boho", "Minimalist"], "names": ["Vase", "Wall Clock", "Desk Lamp", "Rug"], "min_p": 350, "max_p": 15000},
    "fitness": {"prefixes": ["Hex", "Pro", "Heavy Duty"], "names": ["Dumbbell Set", "Yoga Mat", "Resistance Bands", "Kettlebell"], "min_p": 300, "max_p": 8000},
    "books": {"prefixes": ["The Art of", "Mastering", "History of"], "names": ["Deep Learning", "Data Structures", "Quantum Physics", "Financial Freedom"], "min_p": 200, "max_p": 4500},
    "automotive": {"prefixes": ["DriveX", "Turbo", "HD"], "names": ["Dash Cam", "Car Vacuum", "Ceramic Coating", "Phone Mount"], "min_p": 150, "max_p": 9000},
    "toys": {"prefixes": ["Galactic", "Speed", "Classic"], "names": ["Building Blocks", "RC Car", "3x3 Puzzle Cube", "Board Game"], "min_p": 190, "max_p": 12000},
    "groceries": {"prefixes": ["Organic", "Pure", "Raw"], "names": ["Coffee Beans", "Green Tea", "Olive Oil", "Almond Butter"], "min_p": 100, "max_p": 2500}
}

# No @st.cache_data decoration here -> Prevents silent disk serialization crashes
def build_live_catalog():
    raw_data = []
    product_id = 1
    rng = np.random.default_rng(seed=42)
    
    for category in CATEGORIES:
        arch = category_archetypes[category]
        # Generating 200 items per category smoothly straight into memory
        for _ in range(200):
            pfx = rng.choice(arch["prefixes"])
            nm = rng.choice(arch["names"])
            full_name = f"{pfx} {nm}"
            price = int(rng.uniform(arch["min_p"], arch["max_p"]))
            rating = round(float(rng.normal(loc=4.3, scale=0.25)), 1)
            rating = max(1.0, min(5.0, rating))
            
            raw_data.append([product_id, full_name, category, price, rating])
            product_id += 1
            
    return pd.DataFrame(raw_data, columns=["id", "name", "category", "price", "rating"])

products = build_live_catalog()
MAX_GLOBAL_PRICE = 95000.0  # Set as static scalar value
FEATURE_DIM = 12

def build_vectors_fixed(df_subset):
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

def get_candidates(user_vector, category_choice, max_budget, blacklist, search_query=""):
    query_str = str(search_query).strip()
    
    if query_str:
        filter_mask = (
            (products["name"].astype(str).str.contains(query_str, case=False, na=False)) &
            (products["price"] <= max_budget) &
            (~products["id"].isin(blacklist))
        )
    else:
        filter_mask = (
            (products["category"] == category_choice) & 
            (products["price"] <= max_budget) & 
            (~products["id"].isin(blacklist))
        )
        
    filtered_products = products[filter_mask].copy()
    if filtered_products.empty:
        return filtered_products, np.empty((0, FEATURE_DIM), dtype=np.float32)
    
    filtered_vectors = build_vectors_fixed(filtered_products)
    sims = cosine_similarity([user_vector], filtered_vectors)[0]
    
    # 🌟 Performance improvement: Displaying top 2 items keeps the web canvas lightning fast
    actual_k = min(2, len(filtered_products))
    idx = np.argsort(sims)[-actual_k:]
    
    return filtered_products.iloc[idx].copy(), filtered_vectors[idx]

# ==========================================
# 2. THE DEEP RANKING AGENT (PyTorch)
# ==========================================
class DeepRanker(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32), # Sized down layers slightly to minimize memory footprint
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# Avoid disk caching here as well to protect restricted cloud sandbox sessions
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
# 3. LIGHT RESILIENT USER INTERFACE
# ==========================================
st.title("🤖 Self-Learning AI Recommendation Agent")
st.caption("Active Zero-Cache Engine Processing 2,000 SKUs Safely In-Memory.")

if "blacklist" not in st.session_state:
    st.session_state.blacklist = set()

if "current_recommendations" not in st.session_state:
    st.session_state.current_recommendations = None

search_input = st.text_input("🔍 Search for a specific product keyword (e.g., 'iPhone', 'Laptop', 'Runners')", value="")

st.sidebar.header("🎯 Set Your Agent Preferences")
user_cat = st.sidebar.selectbox("Fallback Category", [c.replace("_", " ").title() for c in CATEGORIES])
selected_category_key = user_cat.lower().replace(" ", "_")

user_budget = st.sidebar.slider("Maximum Budget (₹)", min_value=100, max_value=100000, value=50000, step=1000)
user_rating = st.sidebar.slider("Minimum Desired Rating", min_value=1.0, max_value=5.0, value=4.0, step=0.1)

if st.sidebar.button("🧹 Clear Dislike Blacklist"):
    st.session_state.blacklist.clear()
    st.session_state.current_recommendations = None
    st.rerun()

if st.button("🧠 Search & Compute Next Best Action", type="primary", use_container_width=True):
    u_vector = parse_user_input(selected_category_key, user_budget, user_rating)
    candidates, vectors = get_candidates(u_vector, selected_category_key, user_budget, st.session_state.blacklist, search_query=search_input)
    
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
        st.error("❌ No matching products found. Try adjusting your keyword or increasing budget limits.")
    else:
        ranked_df, vectors_used = st.session_state.current_recommendations
        st.subheader("💡 Matching Results & Neural Ranking")
        
        with st.form("feedback_form"):
            feedback_dict = {}
            
            for i, (idx, row) in enumerate(ranked_df.iterrows()):
                col_item, col_feed = st.columns([3, 1])
                with col_item:
                    st.info(f"**{row['name']}** ({row['category'].upper()}) \n💰 Price: ₹{row['price']:,} | ⭐ Rating: {row['rating']} | 🕸️ Current Layer Score: `{round(row['score'], 4)}`")
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
                    with st.spinner("Executing backpropagation layers..."):
                        dynamic_train_step(training_features, training_labels)
                    st.success("🤖 Optimization Complete!")
                    st.session_state.current_recommendations = None
                    st.rerun()
                else:
                    st.warning("Please provide feedback on at least one item to trigger training.")
else:
    st.write("### 👈 Enter a search keyword or change preference configurations to query the AI agent pipeline.")
