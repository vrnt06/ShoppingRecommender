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
    # One-hot encode category (creates 2 columns: category_electronics, category_footwear)
    df = pd.get_dummies(df, columns=["category"])
    # Features left: rating (1), price_norm (1), category_electronics (1), category_footwear (1) = 4 Features Total
    return df.drop(["id", "name", "price"], axis=1).values

# Derive feature dimensions dynamically to prevent ValueError shape mismatches
PRODUCT_VECTORS = build_vectors()
FEATURE_DIM = PRODUCT_VECTORS.shape[1] # Automatically evaluates to 4

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

# Cache resource ensures the model instance persists across user interactions
@st.cache_resource
def init_model(input_dim):
    model = Ranker(input_dim)
    if os.path.exists("model.pt"):
        try:
            model.load_state_dict(torch.load("model.pt", map_location=torch.device('cpu')))
        except Exception:
            pass # Fallback to random weights if file is missing/corrupted
    return model

model = init_model(FEATURE_DIM)

def rank_products(vectors):
    model.eval() 
    with torch.no_grad():
        scores = model(torch.tensor(vectors).float()).numpy().flatten()
    return scores

# ==========================================
# 3. TRAINING ENGINE
# ==========================================
def train_model():
    # Synthetic training data aligned perfectly with the feature dimension
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

    torch.save(model.state_
