import sqlite3
import numpy as np

DB_FILE = "inventory.db"

CATEGORIES = [
    "electronics", "footwear", "clothing", "beauty", "home_decor",
    "fitness", "books", "automotive", "toys", "groceries"
]

# Real-world brands and product types to procedurally cross-multiply into 200 distinct real items per category
seed_matrices = {
    "electronics": {
        "brands": ["Apple", "Samsung", "Sony", "Dell", "HP", "ASUS", "Lenovo", "Logitech", "LG", "Bose", "Anker", "Razer", "Sennheiser", "Corsair", "TP-Link"],
        "items": ["iPhone", "Galaxy S Ultra", "OLED Monitor", "Wireless Headphones", "Mechanical Keyboard", "Gaming Mouse", "UltraBook Laptop", "Tablet Pro", "4K Smart TV", "Bluetooth Speaker", "Power Bank 20K", "Noise Cancelling Earbuds", "Wi-Fi 7 Router", "External SSD 1TB", "Webcam 1080p"],
        "min_p": 1500, "max_p": 160000
    },
    "footwear": {
        "brands": ["Nike", "Adidas", "Puma", "Asics", "New Balance", "Skechers", "Reebok", "Under Armour", "Timberland", "Clarks", "Crocs", "Birkenstock", "Vans", "Converse", "Woodland"],
        "items": ["Air Max Sneakers", "Ultraboost Running Shoes", "Nitro Trail Runners", "Gel-Kayano Stability Shoes", "Classic Lifestyle Sneakers", "Walking Comfort Shoes", "Retro Court Shoes", "Waterproof Hiking Boots", "Leather Chukka Boots", "Classic Clogs", "Two-Strap Sandals", "Canvas High-Tops", "Skate Leather Shoes", "Slip-On Loafers", "Athletic Training Shoes"],
        "min_p": 1200, "max_p": 22000
    },
    "clothing": {
        "brands": ["Levi's", "Patagonia", "The North Face", "Nike", "Adidas", "Uniqlo", "Ralph Lauren", "Tommy Hilfiger", "Calvin Klein", "Columbia", "Carhartt", "Zara", "H&M", "Arc'teryx", "Under Armour"],
        "items": ["Slim Fit Jeans", "Torrentshell Rain Jacket", "Nuptse Down Parka", "Fleece Pullover Hoodie", "Tiro Track Pants", "AIRism Cotton Tee", "Pique Cotton Polo", "Oxford Button-Down Shirt", "Classic Chino Pants", "Full-Zip Fleece Jacket", "Rugged Work Jacket", "Cargo Utility Pants", "Merino Wool Sweater", "Windbreaker Jacket", "Thermal Base Layer"],
        "min_p": 700, "max_p": 35000
    },
    "beauty": {
        "brands": ["COSRX", "The Ordinary", "CeraVe", "La Roche-Posay", "Laneige", "Paula's Choice", "Estée Lauder", "Clinique", "Kiehl's", "Glow Recipe", "Dior", "Chanel", "YSL", "Olaplex", "Mac"],
        "items": ["Snail Mucin Essence", "Niacinamide Serum", "Moisturizing Cream", "SPF 50+ Sunscreen", "Lip Sleeping Mask", "2% BHA Liquid Exfoliant", "Advanced Night Repair", "Hydrating Auto-Replenisher", "Facial Fuel Cleanser", "Watermelon Glow Toner", "Sauvage Eau de Parfum", "Bleu de Chanel Intense", "Radiant Creamy Concealer", "Matte Bullet Lipstick", "Hair Perfector Bond Repair"],
        "min_p": 400, "max_p": 15000
    },
    "home_decor": {
        "brands": ["Philips Hue", "IKEA", "Bath & Body Works", "Marshall", "Safavieh", "Umbra", "Zinus", "Casper", "West Elm", "Home Centre", "Pottery Barn", "Dyson", "Mueller", "Target", "Wayfair"],
        "items": ["Smart LED Starter Kit", "Minimalist Wall Clock", "Scented Triple-Wick Candle", "Abstract Canvas Wall Art", "Woven Cotton Table Runner", "Ceramic Flower Vase Set", "Faux Fur Throw Blanket", "Magnetic Moon Desk Lamp", "Floating Wooden Shelves", "Himalayan Pink Salt Lamp", "Velvet Accent Armchair", "Blended Area Rug", "Geometric Desktop Planter", "Studio Writing Desk", "Pure Cool Purifying Fan"],
        "min_p": 500, "max_p": 45000
    },
    "fitness": {
        "brands": ["Bowflex", "Manduka", "TRX", "Optimum Nutrition", "Myprotein", "Theragun", "Fitbit", "Garmin", "Rogue", "Peloton", "Concept2", "Hydro Flask", "BlenderBottle", "Gymshark", "Under Armour"],
        "items": ["SelectTech Adjustable Dumbbells", "Pro Yoga Mat 6mm", "Suspension Training Kit", "Gold Standard Whey 2kg", "Impact Whey Isolate", "Deep Tissue Massager Gun", "Smart Health Tracker", "GPS Running Smartwatch", "Olympic Steel Barbell", "Indoor Stationary Bike", "Ergonomic Rowing Machine", "Wide Mouth Water Bottle", "Classic Shaker Cup", "Zip-Up Track Jacket", "Compression Training Shorts"],
        "min_p": 300, "max_p": 180000
    },
    "books": {
        "brands": ["O'Reilly Media", "MIT Press", "Penguin", "HarperCollins", "Random House", "Simon & Schuster", "Pearson", "John Wiley", "Oxford Press", "Cambridge University", "Addison-Wesley", "No Starch Press", "Manning", "Packt", "Bloomsbury"],
        "items": ["Designing Data-Intensive Applications", "Deep Learning Foundations", "Atomic Habits Guide", "The Lean Startup Handbook", "Zero to One Masterclass", "Clean Code Engineering", "Introduction to Algorithms", "The Intelligent Investor Edition", "Thinking Fast and Slow Matrix", "Sapiens A Brief History", "Principles for Success", "The Psychology of Money", "Continuous Delivery Frameworks", "The Pragmatic Programmer", "Grokking Algorithms Blueprint"],
        "min_p": 250, "max_p": 8000
    },
    "automotive": {
        "brands": ["Vantrue", "70mai", "Anker Roav", "Chemical Guys", "Meguiar's", "NOCO", "Baseus", "Michelin", "Philips", "Bosch", "Spigen", "Garmin", "Armor All", "ThisWorx", "WeatherTech"],
        "items": ["3-Channel Dash Cam", "Dual Lens Smart Dashcam", "SmartCharge FM Transmitter", "Complete Car Wash Kit", "Premium Liquid Wax", "1000A Battery Jump Starter", "Portable Tyre Inflator", "Digital Pressure Gauge", "X-tremeVision Headlight Bulbs", "ClearMax Wiper Blades", "Magnetic Car Phone Mount", "GPS Smart Navigator", "Interior Glass Wipes Tube", "High Power Car Vacuum", "Laser Measured Floor Mats"],
        "min_p": 300, "max_p": 35000
    },
    "toys": {
        "brands": ["LEGO", "Hasbro", "Mattel", "Rubik's", "DJI Ryze", "Nerf", "Hot Wheels", "Barbie", "Funko", "Fisher-Price", "Ravensburger", "Melissa & Doug", "Play-Doh", "Sphero", "Exploding Kittens"],
        "items": ["Star Wars Building Kit", "Technic Supercar Set", "Catan Strategy Board Game", "Monopoly Ultimate Banking", "Uno Flip Card Edition", "Bluetooth 3x3 Smart Cube", "Tello Mini Drone", "Commander Rotating Blaster", "20-Car Gift Pack Assortment", "Dreamhouse Playset", "Marvel Collectible Figure", "Smart Stages Interactive Puppy", "Villainous Strategy Expansion", "Wooden Engineering Blocks", "Modeling Compound 24-Pack"],
        "min_p": 150, "max_p": 85000
    },
    "groceries": {
        "brands": ["Blue Tokai", "Organic India", "Borges", "Happilo", "Pintola", "Lindt", "Quaker", "Nature's Nectar", "Urban Platter", "Daawat", "Kellogg's", "Saffola", "Hershey's", "Lotus Biscoff", "Nutella"],
        "items": ["Roasted Coffee Beans 250g", "Tulsi Green Tea Bags", "Extra Virgin Olive Oil 1L", "Premium California Almonds", "All-Natural Peanut Butter", "Excellence Dark Chocolate Bar", "Rolled Oats Whole Grain", "Raw Himalayan Honey", "Organic White Chia Seeds", "Super Basmati Rice 5kg", "Corn Flakes Breakfast Cereal", "Blended Refined Cooking Oil", "Natural Unsweetened Cocoa", "Caramelized Biscuit Spread", "Hazelnut Cocoa Spread Jar"],
        "min_p": 50, "max_p": 4500
    }
}

def seed_complete_database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Refresh table to clear outdated baseline sets
    cursor.execute("DROP TABLE IF EXISTS products")
    cursor.execute("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            rating REAL NOT NULL
        )
    """)
    
    rng = np.random.default_rng(seed=101)
    bulk_insert_buffer = []
    
    print("🚀 Starting generation of 2,000 distinct real-world items...")
    
    for category in CATEGORIES:
        matrix = seed_matrices[category]
        generated_count = 0
        
        # Keep picking structural components until we hit exactly 200 items for this specific category
        while generated_count < 200:
            brand = rng.choice(matrix["brands"])
            item = rng.choice(matrix["items"])
            
            # Append a clean serial, modifier, or generation suffix to guarantee string unique variation
            modifier_suffix = rng.choice(["Pro", "Max", "Ultra", "Elite", "Classic", "Series II", "v2", "Edition", "Premium", "Advanced"])
            full_name = f"{brand} {item} ({modifier_suffix})"
            
            # Check for duplicate string mutations to keep data distribution perfect
            if any(full_name == row[0] for row in bulk_insert_buffer):
                continue
                
            price = int(rng.uniform(matrix["min_p"], matrix["max_p"]))
            rating = round(float(rng.normal(loc=4.4, scale=0.25)), 1)
            rating = max(1.0, min(5.0, rating))
            
            bulk_insert_buffer.append((full_name, category, price, rating))
            generated_count += 1
            
        print(f"✅ Generated {generated_count} products for: {category.upper()}")
        
    cursor.executemany("INSERT INTO products (name, category, price, rating) VALUES (?, ?, ?, ?)", bulk_insert_buffer)
    conn.commit()
    
    cursor.execute("SELECT COUNT(*), category FROM products GROUP BY category")
    print("\n📊 Database verification report summary:")
    for row in cursor.fetchall():
        print(f" -> Category '{row[1]}': {row[0]} rows active.")
        
    conn.close()
    print("\n🎉 'inventory.db' generated successfully with 2,000 rows. Ready to run app.py!")

if __name__ == "__main__":
    seed_complete_database()
