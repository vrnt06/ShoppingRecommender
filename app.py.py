import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim

st.set_page_config(page_title="Enterprise AI Agent", page_icon="🤖", layout="wide")

# ==========================================
# 1. GENERATING ENTERPRISE DATA (100 SKUs)
# ==========================================
# Define 10 distinct retail categories
CATEGORIES = [
    "electronics", "footwear", "clothing", "beauty", "home_decor",
    "fitness", "books", "automotive", "toys", "groceries"
]

# High-fidelity baseline data used to generate 10 unique items per category programmatically
category_baselines = {
    "electronics": [("iPhone 13", 60000), ("Samsung S21", 50000), ("Dell Laptop", 70000), ("Sony Headphones", 15000), ("Apple iPad", 55000), ("Logitech Mouse", 8500), ("Smart Watch", 12000), ("4K Monitor", 25000), ("Mechanical Keyboard", 6000), ("GoPro Hero", 35000)],
    "footwear": [("Nike Shoes", 5000), ("Adidas Sneakers", 4500), ("Puma Runners", 3500), ("Bata Formals", 2900), ("Crocs Clogs", 3900), ("Asics Gel", 14000), ("Woodland Boots", 6000), ("Skechers Walkers", 5500), ("Reebok Classics", 4000), ("Flip Flops", 1200)],
    "clothing": [("Denim Jacket", 4000), ("Slim Shirt", 2500), ("Oversized Tee", 1500), ("Chino Trousers", 2200), ("Fleece Hoodie", 4900), ("Polo T-Shirt", 3500), ("Formal Blazer", 6500), ("Cargo Pants", 2800), ("Windbreaker", 4500), ("Puffer Vest", 5000)],
    "beauty": [("Face Serum", 1200), ("Moisturizer", 800), ("Matte Lipstick", 900), ("Sunscreen SPF50", 650), ("Charcoal Facewash", 350), ("Hair Growth Oil", 550), ("Perfume EDP", 4500), ("Night Cream", 1500), ("Body Lotion", 400), ("Eye Palette", 2200)],
    "home_decor": [("Ceramic Vase", 1500), ("Wall Clock", 1800), ("LED Desk Lamp", 2500), ("Scented Candle Set", 1200), ("Throw Blanket", 2000), ("Abstract Canvas", 4500), ("Indoor Planter", 900), ("Boho Rug", 6500), ("Cushion Covers", 600), ("Floating Shelves", 1400)],
    "fitness": [("Dumbbell Set 10kg", 3500), ("Yoga Mat Pro", 1800), ("Resistance Bands", 800), ("Protein Shaker", 600), ("Creatine Monohydrate", 1500), ("Whey Isolate 1kg", 3800), ("Kettlebell 12kg", 2400), ("Jump Rope", 450), ("Gym Gloves", 700), ("Foam Roller", 1100)],
    "books": [("Sci-Fi Novel", 450), ("Python Deep Learning", 3500), ("Biography of Musk", 790), ("Financial Freedom Guide", 550), ("Historical Fiction", 490), ("Self-Help Bestseller", 390), ("Hardcover Atlas", 2500), ("Cookbook Masterclass", 1800), ("Startup Playbook", 690), ("Art History Guide", 1600)],
    "automotive": [("Dash Cam Pro", 5500), ("Car Vacuum Cleaner", 2200), ("Microfiber Towels", 400), ("Ceramic Coating Spray", 950), ("Phone Mount", 600), ("Seat Cushion Gel", 1500), ("Car Air Purifier", 2500), ("Jumper Cables", 1200), ("Tyre Inflator Digital", 3200), ("All-Weather Floor Mats", 4000)],
    "toys": [("Lego Star Wars Set", 8500), ("RC Monster Truck", 3500), ("Rubiks Cube 3x3", 400), ("Board Game Strategy", 2800), ("Plush Teddy Bear", 1200), ("Drawing Tablet Toy", 1500), ("Action Figure", 1800), ("Wooden Puzzle Block", 900), ("Water Gun blaster", 1100), ("Diecast Model Car", 2200)],
    "groceries": [("Premium Coffee Beans", 1200), ("Organic Green Tea", 450), ("Extra Virgin Olive Oil", 1400), ("Almond Butter", 650), ("Dark Chocolate 85%", 300), ("Rolled Oats 1kg", 400), ("Raw Honey", 500), ("Mixed Roasted Nuts", 950), ("Chia Seeds Pro", 350), ("Basmati Rice 5kg", 1100)]
}

# Construct the DataFrame programmatically
raw_data = []
product_id = 1

for category, items in category_baselines.items():
    # Loop generates exactly 10 distinct review configurations per segment
    for i, (name, base_price) in enumerate(items):
        # Apply deterministic variance to ratings and configurations based on index
        rating = round(4.0 + (i % 10) * 0.1, 1) if i % 2 == 0 else round(4.9 - (i % 10) * 0.1, 1)
        raw_data.append([product_id, name, category, base_price, rating])
        product_id += 1

products = pd.DataFrame(raw_data, columns=["id", "name", "category", "price", "rating"])

def build_vectors(df_source=products):
    df = df_source.copy()
    df["price_norm"] = df["price"] / products["price"].max()
    
    # One-hot encode categories safely based on the global categorical spectrum
    df = pd.get_dummies(df, columns=["category"], dtype=int)
    
    # Enforce static column integrity during sliced index lookups
    for cat in [f"category_{c}" for c in CATEGORIES]:
        if cat not in df.columns:
            df[cat] = 0
            
    feature_cols = ["rating", "price_norm"] + [f"category_{c}" for c in CATEGORIES]
    return df[feature_cols].values.astype(np.float32)

PRODUCT_VECTORS = build_vectors()
FEATURE_DIM = PRODUCT_VECTORS.shape[1]  # Evaluates to 12 Dimensions

def parse_user_input(category_choice, max_budget, preferred_rating):
    price_norm = max_budget / products["price"].max()
    vector = [preferred_rating, price_norm]
    
    # Dynamic one-hot generation for the 10 structural segments
    for cat in CATEGORIES:
        vector.append(1.0 if cat == category_choice else 0.0)
        
    return np.array(vector, dtype=np.float32)

def get_candidates(user_vector, category_choice, top_k=3):
    """Stage 1: Strict Category Isolation Shielding"""
    category_mask = products["category"] == category_choice
    filtered_products = products[category_mask].copy()
    
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
    for _ in range(25): # Increased epochs slightly to process higher dimensional data paths
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
st.caption("10 Categories × 10 Items Vector Embedding Pipeline with Implicit User Backpropagation Loops.")

if "current_recommendations" not in st.session_state:
    st.session_state.current_recommendations = None

st.sidebar.header("🎯 Set Your Agent Preferences")
user_cat = st.sidebar.selectbox("Preferred Category", [c.replace("_", " ").title() for c in CATEGORIES])
# Map readable name back to technical key string
selected_category_key = user_cat.lower().replace(" ", "_")

user_budget = st.sidebar.slider("Maximum Budget (₹)", min_value=300, max_value=100000, value=25000, step=500)
user_rating = st.sidebar.slider("Minimum Desired Rating", min_value=1.0, max_value=5.0, value=4.2, step=0.1)

if st.sidebar.button("🧠 Compute Next Best Action", use_container_width=True):
    u_vector = parse_user_input(selected_category_key, user_budget, user_rating)
    candidates, vectors = get_candidates(u_vector, selected_category_key)
    
    agent_nn.eval()
    with torch.no_grad():
        scores = agent_nn(torch.tensor(vectors).float()).numpy().flatten()
    
    candidates["score"] = scores
    ranked_output = candidates.sort_values(by="score", ascending=False)
    st.session_state.current_recommendations = (ranked_output, vectors)

if st.session_state.current_recommendations is not None:
    ranked_df, vectors_used = st.session_state.current_recommendations
    
    st.subheader(f"💡 Agent Recommendations: {user_cat}")
    
    with st.form("feedback_form"):
        feedback_dict = {}
        
        for idx, row in ranked_df.iterrows():
            col_item, col_feed = st.columns([3, 1])
            with col_item:
                st.info(f"**{row['name']}** \n💰 Price: ₹{row['price']} | ⭐ Rating: {row['rating']} | 🕸️ Current Layer Score: `{round(row['score'], 4)}`")
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
            
            if training_features:
                with st.spinner("Executing backpropagation layers..."):
                    dynamic_train_step(training_features, training_labels)
                st.success("🤖 Optimization Complete! Neural weights updated directly from your decisions.")
                st.session_state.current_recommendations = None
                st.rerun()
            else:
                st.warning("Please provide feedback on at least one item to trigger training.")
else:
    st.write("### 👈 Adjust preferences on the sidebar and click **Compute Next Best Action** to initiate the pipeline.")
