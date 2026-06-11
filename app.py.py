import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim

st.set_page_config(page_title="Enterprise AI Agent (Fixed)", page_icon="🤖", layout="wide")

# ==========================================
# 1. PROCEDURAL KNOWLEDGE BASE (2,000+ SKUs)
# ==========================================
CATEGORIES = [
    "electronics", "footwear", "clothing", "beauty", "home_decor",
    "fitness", "books", "automotive", "toys", "groceries"
]

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
def generate_large_catalog():
    raw_data = []
    product_id = 1
    rng = np.random.default_rng(seed=42)
    
    for category in CATEGORIES:
        arch = category_archetypes[category]
        for _ in range(205):  # Exactly 205 items per category
            pfx = rng.choice(arch["prefixes"])
            nm = rng.choice(arch["names"])
            full_name = f"{pfx} {nm}"
            price = int(rng.uniform(arch["min_p"], arch["max_p"]))
            rating = round(float(rng.normal(loc=4.3, scale=0.3)), 1)
            rating = max(1.0, min(5.0, rating))
            
            raw_data.append([product_id, full_name, category, price, rating])
            product_id += 1
            
    return pd.DataFrame(raw_data, columns=["id", "name", "category", "price", "rating"])

products = generate_large_catalog()

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
FEATURE_DIM = PRODUCT_VECTORS.shape[1]

def parse_user_input(category_choice, max_budget, preferred_rating):
    price_norm = max_budget / products["price"].max()
    vector = [preferred_rating, price_norm]
    for cat in CATEGORIES:
        vector.append(1.0 if cat == category_choice else 0.0)
    return np.array(vector, dtype=np.float32)

def get_candidates(user_vector, category_choice, max_budget, blacklist, search_query="", top_k=3):
    """Stage 1: Safe Filter Pipeline with Type-Casting Protection"""
    # Defensive casting to string types to prevent engine evaluation halts
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
        predictions = agent_nn
