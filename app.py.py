import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim

# Set page configuration
st.set_page_config(page_title="FAANG Recommender", page_icon="🔥", layout="wide")

# ==========================================
# 1. DATA & CANDIDATE GENERATION
# ==========================================
products = pd.DataFrame([
    [1, "iPhone 13", "electronics", 60000, 4.7],
    [2, "Samsung S21", "electronics", 50000, 4.5],
    [3, "Nike Shoes", "footwear", 5000, 4.3],
    [4, "Adidas Sneakers", "footwear", 4500, 4.4],
    [5, "Dell Laptop", "electronics", 70000, 4.6],
], columns=["id", "name", "category", "price", "rating"])

def build_vectors():
    df = products.copy()
    # Normalize price
    df["price_norm"] = df["price"] / df["price"].max()
    
    # FIX: Explicitly set dtype=int to prevent boolean (True/False) generation
    df = pd.get_dummies(df, columns=["category"], dtype=int)
    
    # FIX: Force cast the entire numpy matrix to float32 to prevent object_ type issues in PyTorch
    return df.drop(["id", "name", "price"], axis=1).values.astype(np.float32)

# Derive feature dimensions dynamically to keep system matrices perfectly aligned
PRODUCT_VECTORS = build_vectors()
FEATURE_DIM = PRODUCT_VECTORS.shape[1] # Evaluates cleanly to 4

def get_candidates(user_vector, top_k=3):
    sims = cosine_similarity([user_vector], PRODUCT_VECTORS)[0]
    idx = np.argsort(sims)[-top_k:]
    return products.iloc[idx].copy(), PRODUCT_VECTORS[idx]

# ==========================================
# 2. RANKING MODEL (NEURAL NETWORK)
# ==========================================
class Ranker(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Cache resource ensures the model layer weights persist across script reruns
@st.cache_resource
def init_model(input_dim):
    model = Ranker(input_dim)
    if os.path.exists("model.pt"):
        try:
            model.load_state_dict(torch.load("model.pt", map_location=torch.device('cpu')))
        except Exception:
            pass # Fallback to random weights if file is unreadable
    return model

model = init_model(FEATURE_DIM)

def rank_products(vectors):
    model.eval() 
    with torch.no_grad():
        # Clean mapping from float32 numpy arrays directly into the network layers
        scores = model(torch.tensor(vectors).float()).numpy().flatten()
    return scores

# ==========================================
# 3. TRAINING ENGINE
# ==========================================
def train_model():
    # Synthetic training tensors configured to perfectly match the feature space
    X_train = torch.rand((10, FEATURE_DIM))
    y_train = torch.randint(0, 2, (10, 1)).float()
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()

    model.train()
    for epoch in range(50):
        pred = model(X_train)
        loss = loss_fn(pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "model.pt")

# ==========================================
# 4. STREAMLIT UI
# ==========================================
st.title("🔥 FAANG-Level Recommender System")
st.markdown("---")

# Generate a consistent session-cached user vector that mirrors the exact feature dimension
if "user_vector" not in st.session_state:
    st.session_state.user_vector = np.random.rand(FEATURE_DIM).astype(np.float32)

# Grid Layout splitting interactions
col1, col2 = st.columns(2)

with col1:
    st.subheader("💡 Discover Items")
    if st.button("✨ Get Recommendations", use_container_width=True):
        # Stage 1: Candidate Generation (Cosine Similarity Matching)
        candidates, vectors = get_candidates(st.session_state.user_vector)

        # Stage 2: Heavy Ranking (PyTorch Neural Network Scoring)
        scores = rank_products(vectors)
        candidates["score"] = scores
        
        # Sort and display results
        ranked = candidates.sort_values(by="score", ascending=False)
        
        st.write("### Recommended Items (Ranked):")
        for _, row in ranked.iterrows():
            st.info(f"**{row['name']}** \n\n 💰 Price: ₹{row['price']} | ⭐ Rating: {row['rating']} \n\n 🎯 Pipeline Score: `{round(row['score'], 4)}`")

with col2:
    st.subheader("⚙️ Model Operations")
    st.write("Retrain the neural network in-memory to simulate real-time reinforcement training workflows.")
    if st.button("🔄 Retrain Neural Network", use_container_width=True):
        with st.spinner("Retraining PyTorch Network layers..."):
            train_model()
        st.success("Model successfully retrained! Updated weights are live.")
