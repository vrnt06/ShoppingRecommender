import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim

st.set_page_config(page_title="Enterprise AI Agent (Production Safe)", page_icon="🤖", layout="wide")

# Wrap everything in a broad global try-except block to force errors onto the UI if they occur
try:
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
            for _ in range(205):
                pfx = rng.choice(arch["prefixes"])
