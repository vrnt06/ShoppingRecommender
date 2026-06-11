"""
seed_db.py — Inventory database seeder.
Generates 2,000 distinct products across 10 categories and writes them to inventory.db.

Run once before launching app.py:
    python seed_db.py
"""

import sqlite3
import numpy as np

DB_FILE = "inventory.db"

CATEGORIES = [
    "electronics", "footwear", "clothing", "beauty", "home_decor",
    "fitness", "books", "automotive", "toys", "groceries",
]

# Each category defines brands, product lines, price range, and rating distribution.
SEED_MATRICES: dict[str, dict] = {
    "electronics": {
        "brands": ["Apple", "Samsung", "Sony", "Dell", "HP", "ASUS", "Lenovo", "Logitech", "LG", "Bose",
                   "Anker", "Razer", "Sennheiser", "Corsair", "TP-Link"],
        "items": ["iPhone", "Galaxy S Ultra", "OLED Monitor", "Wireless Headphones", "Mechanical Keyboard",
                  "Gaming Mouse", "UltraBook Laptop", "Tablet Pro", "4K Smart TV", "Bluetooth Speaker",
                  "Power Bank 20K", "Noise Cancelling Earbuds", "Wi-Fi 7 Router", "External SSD 1TB", "Webcam 1080p"],
        "min_p": 1_500, "max_p": 160_000, "rating_loc": 4.3, "rating_scale": 0.30,
    },
    "footwear": {
        "brands": ["Nike", "Adidas", "Puma", "Asics", "New Balance", "Skechers", "Reebok", "Under Armour",
                   "Timberland", "Clarks", "Crocs", "Birkenstock", "Vans", "Converse", "Woodland"],
        "items": ["Air Max Sneakers", "Ultraboost Running Shoes", "Nitro Trail Runners", "Gel-Kayano Stability Shoes",
                  "Classic Lifestyle Sneakers", "Walking Comfort Shoes", "Retro Court Shoes", "Waterproof Hiking Boots",
                  "Leather Chukka Boots", "Classic Clogs", "Two-Strap Sandals", "Canvas High-Tops",
                  "Skate Leather Shoes", "Slip-On Loafers", "Athletic Training Shoes"],
        "min_p": 1_200, "max_p": 22_000, "rating_loc": 4.4, "rating_scale": 0.25,
    },
    "clothing": {
        "brands": ["Levi's", "Patagonia", "The North Face", "Nike", "Adidas", "Uniqlo", "Ralph Lauren",
                   "Tommy Hilfiger", "Calvin Klein", "Columbia", "Carhartt", "Zara", "H&M", "Arc'teryx", "Under Armour"],
        "items": ["Slim Fit Jeans", "Torrentshell Rain Jacket", "Nuptse Down Parka", "Fleece Pullover Hoodie",
                  "Tiro Track Pants", "AIRism Cotton Tee", "Pique Cotton Polo", "Oxford Button-Down Shirt",
                  "Classic Chino Pants", "Full-Zip Fleece Jacket", "Rugged Work Jacket", "Cargo Utility Pants",
                  "Merino Wool Sweater", "Windbreaker Jacket", "Thermal Base Layer"],
        "min_p": 700, "max_p": 35_000, "rating_loc": 4.3, "rating_scale": 0.28,
    },
    "beauty": {
        "brands": ["COSRX", "The Ordinary", "CeraVe", "La Roche-Posay", "Laneige", "Paula's Choice",
                   "Estée Lauder", "Clinique", "Kiehl's", "Glow Recipe", "Dior", "Chanel", "YSL", "Olaplex", "Mac"],
        "items": ["Snail Mucin Essence", "Niacinamide Serum", "Moisturizing Cream", "SPF 50+ Sunscreen",
                  "Lip Sleeping Mask", "2% BHA Liquid Exfoliant", "Advanced Night Repair", "Hydrating Auto-Replenisher",
                  "Facial Fuel Cleanser", "Watermelon Glow Toner", "Sauvage Eau de Parfum",
                  "Bleu de Chanel Intense", "Radiant Creamy Concealer", "Matte Bullet Lipstick", "Hair Perfector Bond Repair"],
        "min_p": 400, "max_p": 15_000, "rating_loc": 4.5, "rating_scale": 0.22,
    },
    "home_decor": {
        "brands": ["Philips Hue", "IKEA", "Bath & Body Works", "Marshall", "Safavieh", "Umbra", "Zinus",
                   "Casper", "West Elm", "Home Centre", "Pottery Barn", "Dyson", "Mueller", "Target", "Wayfair"],
        "items": ["Smart LED Starter Kit", "Minimalist Wall Clock", "Scented Triple-Wick Candle",
                  "Abstract Canvas Wall Art", "Woven Cotton Table Runner", "Ceramic Flower Vase Set",
                  "Faux Fur Throw Blanket", "Magnetic Moon Desk Lamp", "Floating Wooden Shelves",
                  "Himalayan Pink Salt Lamp", "Velvet Accent Armchair", "Blended Area Rug",
                  "Geometric Desktop Planter", "Studio Writing Desk", "Pure Cool Purifying Fan"],
        "min_p": 500, "max_p": 45_000, "rating_loc": 4.3, "rating_scale": 0.28,
    },
    "fitness": {
        "brands": ["Bowflex", "Manduka", "TRX", "Optimum Nutrition", "Myprotein", "Theragun", "Fitbit",
                   "Garmin", "Rogue", "Peloton", "Concept2", "Hydro Flask", "BlenderBottle", "Gymshark", "Under Armour"],
        "items": ["SelectTech Adjustable Dumbbells", "Pro Yoga Mat 6mm", "Suspension Training Kit",
                  "Gold Standard Whey 2kg", "Impact Whey Isolate", "Deep Tissue Massager Gun",
                  "Smart Health Tracker", "GPS Running Smartwatch", "Olympic Steel Barbell",
                  "Indoor Stationary Bike", "Ergonomic Rowing Machine", "Wide Mouth Water Bottle",
                  "Classic Shaker Cup", "Zip-Up Track Jacket", "Compression Training Shorts"],
        "min_p": 300, "max_p": 180_000, "rating_loc": 4.4, "rating_scale": 0.25,
    },
    "books": {
        "brands": ["O'Reilly Media", "MIT Press", "Penguin", "HarperCollins", "Random House",
                   "Simon & Schuster", "Pearson", "John Wiley", "Oxford Press", "Cambridge University",
                   "Addison-Wesley", "No Starch Press", "Manning", "Packt", "Bloomsbury"],
        "items": ["Designing Data-Intensive Applications", "Deep Learning Foundations", "Atomic Habits Guide",
                  "The Lean Startup Handbook", "Zero to One Masterclass", "Clean Code Engineering",
                  "Introduction to Algorithms", "The Intelligent Investor Edition", "Thinking Fast and Slow Matrix",
                  "Sapiens A Brief History", "Principles for Success", "The Psychology of Money",
                  "Continuous Delivery Frameworks", "The Pragmatic Programmer", "Grokking Algorithms Blueprint"],
        "min_p": 250, "max_p": 8_000, "rating_loc": 4.5, "rating_scale": 0.20,
    },
    "automotive": {
        "brands": ["Vantrue", "70mai", "Anker Roav", "Chemical Guys", "Meguiar's", "NOCO", "Baseus",
                   "Michelin", "Philips", "Bosch", "Spigen", "Garmin", "Armor All", "ThisWorx", "WeatherTech"],
        "items": ["3-Channel Dash Cam", "Dual Lens Smart Dashcam", "SmartCharge FM Transmitter",
                  "Complete Car Wash Kit", "Premium Liquid Wax", "1000A Battery Jump Starter",
                  "Portable Tyre Inflator", "Digital Pressure Gauge", "X-tremeVision Headlight Bulbs",
                  "ClearMax Wiper Blades", "Magnetic Car Phone Mount", "GPS Smart Navigator",
                  "Interior Glass Wipes Tube", "High Power Car Vacuum", "Laser Measured Floor Mats"],
        "min_p": 300, "max_p": 35_000, "rating_loc": 4.3, "rating_scale": 0.28,
    },
    "toys": {
        "brands": ["LEGO", "Hasbro", "Mattel", "Rubik's", "DJI Ryze", "Nerf", "Hot Wheels", "Barbie",
                   "Funko", "Fisher-Price", "Ravensburger", "Melissa & Doug", "Play-Doh", "Sphero", "Exploding Kittens"],
        "items": ["Star Wars Building Kit", "Technic Supercar Set", "Catan Strategy Board Game",
                  "Monopoly Ultimate Banking", "Uno Flip Card Edition", "Bluetooth 3x3 Smart Cube",
                  "Tello Mini Drone", "Commander Rotating Blaster", "20-Car Gift Pack Assortment",
                  "Dreamhouse Playset", "Marvel Collectible Figure", "Smart Stages Interactive Puppy",
                  "Villainous Strategy Expansion", "Wooden Engineering Blocks", "Modeling Compound 24-Pack"],
        "min_p": 150, "max_p": 85_000, "rating_loc": 4.4, "rating_scale": 0.26,
    },
    "groceries": {
        "brands": ["Blue Tokai", "Organic India", "Borges", "Happilo", "Pintola", "Lindt", "Quaker",
                   "Nature's Nectar", "Urban Platter", "Daawat", "Kellogg's", "Saffola", "Hershey's",
                   "Lotus Biscoff", "Nutella"],
        "items": ["Roasted Coffee Beans 250g", "Tulsi Green Tea Bags", "Extra Virgin Olive Oil 1L",
                  "Premium California Almonds", "All-Natural Peanut Butter", "Excellence Dark Chocolate Bar",
                  "Rolled Oats Whole Grain", "Raw Himalayan Honey", "Organic White Chia Seeds",
                  "Super Basmati Rice 5kg", "Corn Flakes Breakfast Cereal", "Blended Refined Cooking Oil",
                  "Natural Unsweetened Cocoa", "Caramelized Biscuit Spread", "Hazelnut Cocoa Spread Jar"],
        "min_p": 50, "max_p": 4_500, "rating_loc": 4.4, "rating_scale": 0.22,
    },
}

MODIFIERS = [
    "Pro", "Max", "Ultra", "Elite", "Classic", "Series II", "v2",
    "Edition", "Premium", "Advanced", "Signature", "Select", "Core", "Lite", "Plus",
]

PRODUCTS_PER_CATEGORY = 200


def _generate_products(rng: np.random.Generator) -> list[tuple[str, str, int, float]]:
    """
    Return a flat list of (name, category, price, rating) tuples.
    Each category gets exactly PRODUCTS_PER_CATEGORY unique entries.
    O(n) duplicate detection via a set — not a linear scan.
    """
    rows: list[tuple[str, str, int, float]] = []

    for category in CATEGORIES:
        mx = SEED_MATRICES[category]
        brands = mx["brands"]
        items = mx["items"]
        min_p, max_p = mx["min_p"], mx["max_p"]
        rating_loc, rating_scale = mx["rating_loc"], mx["rating_scale"]

        seen_names: set[str] = set()
        generated = 0

        while generated < PRODUCTS_PER_CATEGORY:
            name = (
                f"{rng.choice(brands)} "
                f"{rng.choice(items)} "
                f"({rng.choice(MODIFIERS)})"
            )
            if name in seen_names:
                continue

            seen_names.add(name)

            price = int(rng.uniform(min_p, max_p))
            rating = float(np.clip(rng.normal(rating_loc, rating_scale), 1.0, 5.0))
            rating = round(rating, 1)

            rows.append((name, category, price, rating))
            generated += 1

        print(f"  ✓  {category:<12}  {generated} products")

    return rows


def seed_complete_database() -> None:
    rng = np.random.default_rng(seed=101)

    print("Generating products …")
    rows = _generate_products(rng)

    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    cur.executescript("""
        DROP TABLE IF EXISTS products;
        CREATE TABLE products (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            name     TEXT    NOT NULL,
            category TEXT    NOT NULL,
            price    REAL    NOT NULL,
            rating   REAL    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_cat_price ON products (category, price);
    """)

    cur.executemany(
        "INSERT INTO products (name, category, price, rating) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()

    # Verification
    cur.execute("SELECT category, COUNT(*) AS n FROM products GROUP BY category ORDER BY category")
    print("\nRow counts per category:")
    total = 0
    for cat, n in cur.fetchall():
        print(f"  {cat:<14} {n:>4}")
        total += n
    print(f"  {'TOTAL':<14} {total:>4}")

    conn.close()
    print(f"\ninventory.db ready — {total} rows written.")


if __name__ == "__main__":
    seed_complete_database()
