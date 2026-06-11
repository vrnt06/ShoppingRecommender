import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim

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
    df["price_norm"] = df["price"] / df["price"].max()
    # Explicitly use prefix to avoid column matching errors
    df = pd.get_dummies(df, columns=["category"])
    # Return features excluding id, name, and raw price
    return df.drop(["id", "name", "price"], axis=1).values

def get_candidates(user_vector, top_k=3):
    vectors = build_vectors()
    sims = cosine_similarity([user_vector], vectors)[0]
    idx = np.argsort(sims)[-top_k:]
    return products.iloc[idx].copy(), vectors[idx]

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

# Cache resource ensures the model isn't re-instantiated on every button click
@st.cache_resource
def init_model(input_dim=5):
    model = Ranker(input_dim)
    if os.path.exists("model.pt"):
        try:
            model.load_state_dict(torch.load("model.pt", map_location=torch.device('cpu')))
        except Exception:
            pass # Fallback to random weights if model file is corrupted
    return model

model = init_model()

def rank_products(vectors):
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        scores = model(torch.tensor(vectors).float()).numpy().flatten()
    return scores

# ==========================================
# 3. TRAINING ENGINE
# ==========================================
def train_model():
    # Dummy training data 
    X_train = torch.rand((10, 5))
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
st.caption("A Two-Stage Recommendation Pipeline: Cosine Similarity Candidate Generation + PyTorch Ranker NN")

# Generate a consistent fake user vector based on session state so it doesn't shift constantly
if "user_vector" not in st.session_state:
    st.session_state.user_vector = np.random.rand(5)

col1, col2 = st.columns(2)

with col1:
    if st.button("✨ Get Recommendations", use_container_width=True):
        st.subheader("Top Ranked Items")
        
        # 1. Candidate Retrieval Stage
        candidates, vectors = get_candidates(st.session_state.user_vector)

        # 2. Ranking Stage
        scores = rank_products(vectors)
        candidates["score"] = scores
        
        # Sort and Display
        ranked = candidates.sort_values(by="score", ascending=False)
        
        for _, row in ranked.iterrows():
            st.info(f"**{row['name']}** \n💰 Price: ₹{row['price']} | ⭐ Rating: {row['rating']}  \n🎯 AI Ranking Score: `{round(row['score'], 4)}`")

with col2:
    if st.button("⚙️ Retrain Model", use_container_width=True):
        with st.spinner("Retraining PyTorch Neural Network..."):
            train_model()
        st.success("Model retrained and weights updated!")