import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim

st.set_page_config(page_title="Enterprise AI Agent (2K+ SKUs)", page_icon="🤖", layout="wide")

# ==========================================
# 1. PROCEDURAL KNOWLEDGE BASE (2,000+ SKUs)
# ==========================================
CATEGORIES = [
    "electronics", "footwear", "clothing", "beauty", "home_decor",
    "fitness", "books", "automotive", "toys", "groceries"
]

# Baseline structural archetypes used to procedurally seed the data generation engine
category_archetypes = {
    "electronics": {"prefixes": ["Pro", "Ultra", "Max", "Quantum", "Apex"], "names": ["Laptop", "Smartphone", "Headphones", "Monitor", "Smartwatch", "Keyboard", "Tablet", "Speaker"], "min_p": 1500, "max_p": 95000},
    "footwear": {"prefixes": ["Air", "Gel", "Zoom", "Classic", "Trail"], "names": ["Runners", "Sneakers", "Formals", "Boots", "Loafers", "Sandals", "Clogs", "Trainers"], "min_p": 800, "max_p": 18000},
    "clothing": {"prefixes": ["Urban", "Slim Fit", "Oversized", "Premium", "Vintage"], "names": ["Hoodie", "Jacket", "T-Shirt", "Chinos", "Shirt", "Cargo Pants", "Blazer", "Sweater"], "min_p": 500, "max_p": 12000},
    "beauty": {"prefixes": ["Hydra", "Glow", "Matte", "Organic", "Revitalizing"], "names": ["Serum", "Moisturizer", "Lipstick", "Sunscreen", "Facewash", "Face Mask", "Perfume", "Night Cream"], "min_p": 250, "max_p": 6000},
    "home_decor": {"prefixes": ["Nordic", "Boho", "Minimalist", "Rustic", "Luxury"], "names": ["Vase", "Wall Clock", "Desk Lamp", "Scented Candle", "Rug", "Canvas Art", "Planter", "Shelves"], "min_p": 350, "max_p": 15000},
    "fitness": {"prefixes": ["Hex", "Pro", "Heavy Duty", "Ergonomic", "Isolate"], "names": ["Dumbbell Set", "Yoga Mat", "Resistance Bands", "Kettlebell", "Whey Protein", "Jump Rope", "Gym Gloves", "Foam Roller"], "min_p": 300, "max_p": 8000},
    "books": {"prefixes": ["The Art of", "Mastering", "History of", "Introduction to", "The Secret of"], "names": ["Deep Learning", "Data Structures", "Quantum Physics", "Financial Freedom", "Sci-Fi Trilogy", "Macroeconomics", "Biographies", "Creative Writing"], "min_p": 200, "max_p": 4500},
    "automotive": {"prefixes": ["DriveX", "Turbo", "HD", "All-Weather", "Premium"], "names": ["Dash Cam", "Car Vacuum", "Ceramic Coating", "Phone Mount", "Gel Cushion", "Air Purifier", "Tyre Inflator", "Floor Mats"], "min_p": 150, "max_p": 9000},
    "toys": {"prefixes": ["Galactic", "Speed", "Classic", "Brainiac", "Retro"], "names": ["Building Blocks", "RC Car", "3x3 Puzzle Cube", "Board Game", "Plush Toy", "Drawing Tablet", "Action Figure", "Diecast Car"], "min_p": 190, "max_p": 12000},
    "groceries": {"prefixes": ["Organic", "Pure", "Raw", "Roasted", "Artisanal"], "names": ["Coffee Beans", "Green Tea", "Olive Oil", "Almond Butter", "Dark Chocolate", "Rolled Oats", "Wild Honey", "Mixed Nuts"], "min_p": 100, "max_p": 2500}
}

@st.cache_data
def generate_large_catalog(items_per_category=200):
    """Procedurally synthesizes a scaled retail universe containing varied metadata."""
    raw_data = []
    product_id = 1
    
    # State-controlled pseudo-random engine for reproducible matrix dimensions
    rng = np.random.default_rng(seed=42)
    
    for category in CATEGORIES:
        arch = category_archetypes[category]
        for _ in range(items_per_category):
            pfx = rng.choice(arch["prefixes"])
            nm = rng.choice(arch["names"])
            full_name = f"{pfx} {nm}"
            
            # Generate uniform log-scale distribution for realistic pricing models
            price = int(rng.uniform(arch["min_p"], arch["max_p"]))
            # Normal distribution centering ratings tightly around high-quality parameters
            rating = round(float(rng.normal(loc=4.3, scale=0.3)), 1)
            rating = max(1.0, min(5.0, rating)) # Strict clip bounds
            
            raw_data.append([product_id, full_name, category, price, rating])
            product_id += 1
            
    return pd.DataFrame(raw_data, columns=["id", "name", "category", "price", "rating"])

products = generate_large_catalog(items_per_category=205) # Total ~2,050 high-quality products

def build_vectors(df_source=products):
    df = df_source.copy()
    df["price_norm"] = df["price"] / products["price"].max()
    df = pd.get_dummies(df, columns=["category"], dtype=int)
    
    for cat in [f"category_{c}" for c in CATEGORIES]:
        if cat not in df.columns:
            df[cat] = 0
            
    feature_cols = ["rating", "price_norm"] + [f"category_{c}" for c in CATEGORIES]
    return df[feature_cols].values.astype(np.float32)

PRODUCT_VECTORS = build_vectors()
FEATURE_DIM = PRODUCT_VECTORS.shape[1]  # 12 Dimensions

def parse_user_input(category_choice, max_budget, preferred_rating):
    price_norm = max_budget / products["price"].max()
    vector = [preferred_rating, price_norm]
    for cat in CATEGORIES:
        vector.append(1.0 if cat == category_choice else 0.0)
    return np.array(vector, dtype=np.float32)

def get_candidates(user_vector, category_choice, max_budget, blacklist, top_k=3):
    """Stage 1: Matrix Filter (Slicing active structures, pricing caps, and blacklist matrices)"""
    filter_mask = (
        (products["category"] == category_choice) & 
        (products["price"] <= max_budget) & 
        (~products["id"].isin(blacklist))
    )
    filtered_products = products[filter_mask].copy()
    
    if filtered_products.empty:
        return filtered_products, np.array([], dtype=np.float32).reshape(0, FEATURE_DIM)
    
    filtered_vectors = build_vectors(filtered_products)
    sims = cosine_similarity([user_vector], filtered_vectors)[0]
    
    actual_k = min(top_k, len(filtered_products))
    idx = np.argsort(sims)[-actual_k:]
    
    return filtered_products.iloc[idx].copy(), filtered_vectors[idx]

# ==========================================
# 2. THE DEEP RANKING AGENT (PyTorch)
# ==========================================
class DeepRanker(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

@st.cache_resource
def load_agent(input_dim):
    agent = DeepRanker(input_dim)
    if os.path.exists("agent_weights.pt"):
        try:
            agent.load_state_dict(torch.load("agent_weights.pt", map_location=torch.device('cpu')))
        except:
            pass
    return agent

agent_nn = load_agent(FEATURE_DIM)

# ==========================================
# 3. LIVE LEARNING ENGINE
# ==========================================
def dynamic_train_step(features, labels):
    X = torch.tensor(np.array(features)).float()
    y = torch.tensor(np.array(labels)).float().unsqueeze(1)
    
    optimizer = optim.Adam(agent_nn.parameters(), lr=0.05)
    loss_fn = nn.BCELoss()
    
    agent_nn.train()
    for _ in range(25):
        predictions = agent_nn(X)
        loss = loss_fn(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    torch.save(agent_nn.state_dict(), "agent_weights.pt")

# ==========================================
# 4. STREAMLIT INTERFACE
# ==========================================
st.title("🤖 Self-Learning AI Recommendation Agent")
st.caption(f"Enterprise Scaled Universe: {len(products):,} Real-time Generated Products across {len(CATEGORIES)} Functional Segments.")

if "blacklist" not in st.session_state:
    st.session_state.blacklist = set()

if "current_recommendations" not in st.session_state:
    st.session_state.current_recommendations = None

st.sidebar.header("🎯 Set Your Agent Preferences")
user_cat = st.sidebar.selectbox("Preferred Category", [c.replace("_", " ").title() for c in CATEGORIES])
selected_category_key = user_cat.lower().replace(" ", "_")

user_budget = st.sidebar.slider("Maximum Budget (₹)", min_value=100, max_value=100000, value=25000, step=500)
user_rating = st.sidebar.slider("Minimum Desired Rating", min_value=1.0, max_value=5.0, value=4.2, step=0.1)

if st.sidebar.button("🧹 Clear Dislike Blacklist"):
    st.session_state.blacklist.clear()
    st.session_state.current_recommendations = None
    st.sidebar.success("Blacklist cleared!")

if st.sidebar.button("🧠 Compute Next Best Action", use_container_width=True):
    u_vector = parse_user_input(selected_category_key, user_budget, user_rating)
    candidates, vectors = get_candidates(u_vector, selected_category_key, user_budget, st.session_state.blacklist)
    
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
        st.error(f"❌ No eligible items found in **{user_cat}**. They may be filtered out by budget or your dislike blacklist.")
    else:
        ranked_df, vectors_used = st.session_state.current_recommendations
        st.subheader(f"💡 Agent Recommendations: {user_cat}")
        
        with st.form("feedback_form"):
            feedback_dict = {}
            
            for i, (idx, row) in enumerate(ranked_df.iterrows()):
                col_item, col_feed = st.columns([3, 1])
                with col_item:
                    st.info(f"**{row['name']}** \n💰 Price: ₹{row['price']:,} | ⭐ Rating: {row['rating']} | 🕸️ Current Layer Score: `{round(row['score'], 4)}`")
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
                    st.success("🤖 Optimization Complete! Weights updated and disliked items blacklisted.")
                    st.session_state.current_recommendations = None
                    st.rerun()
                else:
                    st.warning("Please provide feedback on at least one item to trigger training.")
else:
    st.write("### 👈 Adjust preferences on the sidebar and click **Compute Next Best Action** to initiate the pipeline.")
