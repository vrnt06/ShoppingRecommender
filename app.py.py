import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim

st.set_page_config(page_title="AI Recommendation Agent", page_icon="🤖", layout="wide")

# ==========================================
# 1. KNOWLEDGE BASE & EMBEDDING PIPELINE
# ==========================================
# High-dimensional catalog
products = pd.DataFrame([
    [1, "iPhone 13", "electronics", 60000, 4.7],
    [2, "Samsung S21", "electronics", 50000, 4.5],
    [3, "Nike Shoes", "footwear", 5000, 4.3],
    [4, "Adidas Sneakers", "footwear", 4500, 4.4],
    [5, "Dell Laptop", "electronics", 70000, 4.6],
], columns=["id", "name", "category", "price", "rating"])

def build_vectors():
    df = products.copy()
    df["price_norm"] = df["price"] / df["price"].max()
    df = pd.get_dummies(df, columns=["category"], dtype=int)
    # Target Features: [rating, price_norm, cat_electronics, cat_footwear]
    return df.drop(["id", "name", "price"], axis=1).values.astype(np.float32)

PRODUCT_VECTORS = build_vectors()
FEATURE_DIM = PRODUCT_VECTORS.shape[1]  # 4

def parse_user_input(category_choice, max_budget, preferred_rating):
    """
    Agent Input Parser: Converts structured user requirements 
    into a mathematically aligned feature vector.
    """
    price_norm = max_budget / products["price"].max()
    cat_electronics = 1 if category_choice == "electronics" else 0
    cat_footwear = 1 if category_choice == "footwear" else 0
    
    # Matches [rating, price_norm, cat_electronics, cat_footwear]
    return np.array([preferred_rating, price_norm, cat_electronics, cat_footwear], dtype=np.float32)

def get_candidates(user_vector, top_k=3):
    """Stage 1: Vector Space Filtering"""
    sims = cosine_similarity([user_vector], PRODUCT_VECTORS)[0]
    idx = np.argsort(sims)[-top_k:]
    return products.iloc[idx].copy(), PRODUCT_VECTORS[idx]

# ==========================================
# 2. THE DEEP RANKING AGENT (PyTorch)
# ==========================================
class DeepRanker(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
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
# 3. LIVE ON-THE-FLY LEARNING ENGINE
# ==========================================
def dynamic_train_step(features, labels):
    """
    Performs real-time backpropagation based on active user metrics.
    Optimizes weights on live production feedback instantly.
    """
    X = torch.tensor(np.array(features)).float()
    y = torch.tensor(np.array(labels)).float().unsqueeze(1)
    
    optimizer = optim.Adam(agent_nn.parameters(), lr=0.05) # Aggressive learning rate for immediate feedback loop
    loss_fn = nn.BCELoss()
    
    agent_nn.train()
    for _ in range(20):  # Quick optimization cycle
        predictions = agent_nn(X)
        loss = loss_fn(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    torch.save(agent_nn.state_dict(), "agent_weights.pt")

# ==========================================
# 4. AGENT STREAMLIT INTERFACE
# ==========================================
st.title("🤖 Self-Learning AI Recommendation Agent")
st.caption("Two-Stage Candidate Generation Engine powered by interactive feedback loops.")

# Session Memory to track user actions
if "interaction_history" not in st.session_state:
    st.session_state.interaction_history = []
if "current_recommendations" not in st.session_state:
    st.session_state.current_recommendations = None

# Sidebar Controls for User Profiles
st.sidebar.header("🎯 Set Your Agent Preferences")
user_cat = st.sidebar.selectbox("Preferred Category", ["electronics", "footwear"])
user_budget = st.sidebar.slider("Maximum Budget (₹)", min_value=5000, max_value=100000, value=50000, step=5000)
user_rating = st.sidebar.slider("Minimum Desired Rating", min_value=1.0, max_value=5.0, value=4.5, step=0.1)

if st.sidebar.button("🧠 Compute Next Best Action", use_container_width=True):
    # Step 1: Parse Input to Vector Space
    u_vector = parse_user_input(user_cat, user_budget, user_rating)
    
    # Step 2: Retrieve Candidates via Cosine Similarity
    candidates, vectors = get_candidates(u_vector)
    
    # Step 3: Run Inference through the Neural Network
    agent_nn.eval()
    with torch.no_grad():
        scores = agent_nn(torch.tensor(vectors).float()).numpy().flatten()
    
    candidates["score"] = scores
    ranked_output = candidates.sort_values(by="score", ascending=False)
    
    # Store state for user reinforcement feedback
    st.session_state.current_recommendations = (ranked_output, vectors)

# Main UI Display Area
if st.session_state.current_recommendations is not None:
    ranked_df, vectors_used = st.session_state.current_recommendations
    
    st.subheader("💡 Agent Recommendations")
    st.write("Review the items below and provide implicit feedback to train the agent's internal network layers.")
    
    # Track actions inside a dynamic form
    with st.form("feedback_form"):
        feedback_dict = {}
        
        for idx, row in ranked_df.iterrows():
            col_item, col_feed = st.columns([3, 1])
            with col_item:
                st.info(f"**{row['name']}** ({row['category'].upper()})  \n💰 Price: ₹{row['price']} | ⭐ Rating: {row['rating']} | 🕸️ Current Layer Score: `{round(row['score'], 4)}`")
            with col_feed:
                # User training inputs
                feedback_dict[idx] = st.radio("Feedback", ["Select Action", "👍 Like / Buy", "👎 Dislike / Ignore"], key=f"feed_{row['id']}")
                
        submitted = st.form_submit_with_warm_start("📥 Process Feedback & Train Neural Layers", use_container_width=True)
        
        if submitted:
            training_features = []
            training_labels = []
            
            # Map user actions directly to mathematical labels (1.0 or 0.0)
            for i, (idx, row) in enumerate(ranked_df.iterrows()):
                action = feedback_dict[idx]
                if action != "Select Action":
                    training_features.append(vectors_used[i])
                    training_labels.append(1.0 if action == "👍 Like / Buy" else 0.0)
            
            if training_features:
                with st.spinner("Executing backpropagation layers..."):
                    dynamic_train_step(training_features, training_labels)
                st.success("🤖 Optimization Complete! Neural weights updated directly from your decisions.")
                st.session_state.current_recommendations = None  # Reset to update views on next action run
                st.rerun()
            else:
                st.warning("Please provide feedback on at least one item to trigger training.")
else:
    st.write("### 👈 Adjust preferences on the sidebar and click **Compute Next Best Action** to initiate the pipeline.")

# Metrics / Observability Dashboard
st.markdown("---")
st.subheader("📊 System Architecture Diagnostics")
col_stat1, col_stat2 = st.columns(2)
with col_stat1:
    st.metric(label="Neural Network Input Matrix Dimension", value=f"[{FEATURE_DIM}] Float32")
with col_stat2:
    st.metric(label="Active Weight Persistence Matrix File", value="agent_weights.pt" if os.path.exists("agent_weights.pt") else "In-Memory Defaults")
