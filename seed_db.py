"""
seed_db.py — Inventory database seeder.
Generates 2,000 distinct products across 10 categories and writes them to inventory.db,
along with associated rich metadata (brand, model, stock, descriptions, Unsplash URLs)
and a robust reviews table with 6,000 generated reviews.

Run once before launching app.py:
    python seed_db.py
"""

import sqlite3
import numpy as np
import hashlib

DB_FILE = "inventory.db"

CATEGORIES = [
    "electronics", "footwear", "clothing", "beauty", "home_decor",
    "fitness", "books", "automotive", "toys", "groceries",
]

SEED_MATRICES = {
    "electronics": {
        "brands": ["Apple", "Samsung", "Sony", "Dell", "HP", "ASUS", "Lenovo", "Logitech", "LG", "Bose",
                   "Anker", "Razer", "Sennheiser", "Corsair", "TP-Link"],
        "items": ["iPhone", "Galaxy S Ultra", "OLED Monitor", "Wireless Headphones", "Mechanical Keyboard",
                  "Gaming Mouse", "UltraBook Laptop", "Tablet Pro", "4K Smart TV", "Bluetooth Speaker",
                  "Power Bank 20K", "Noise Cancelling Earbuds", "Wi-Fi 7 Router", "External SSD 1TB", "Webcam 1080p"],
        "min_p": 1500, "max_p": 160000, "rating_loc": 4.3, "rating_scale": 0.30,
        "images": [
            "photo-1511707171634-5f897ff02aa9", # iPhone
            "photo-1610945265064-0e34e5519bbf", # Galaxy S Ultra
            "photo-1527443224154-c4a3942d3acf", # OLED Monitor
            "photo-1505740420928-5e560c06d30e", # Wireless Headphones
            "photo-1587829741301-dc798b83add3", # Mechanical Keyboard
            "photo-1615663245857-ac93bb7c39e7", # Gaming Mouse
            "photo-1496181130204-755241524eab", # UltraBook Laptop
            "photo-1544244015-0df4b3ffc6b0", # Tablet Pro
            "photo-1593305841991-05c297ba4575", # 4K Smart TV
            "photo-1608043152269-423dbba4e7e1", # Bluetooth Speaker
            "photo-1609081219090-a6d81d3085bf", # Power Bank 20K
            "photo-1590658268037-6bf12165a8df", # Noise Cancelling Earbuds
            "photo-1544197150-b99a580bb7a8", # Wi-Fi 7 Router
            "photo-1618424181497-157f25b6ddd5", # External SSD 1TB
            "photo-1601524909162-be87252be298"  # Webcam 1080p
        ],
        "desc": "An executive-grade {brand} {item} ({modifier}) designed for high performance. Offers stunning visual clarity, lightning-fast response times, and premium durability for discerning users."
    },
    "footwear": {
        "brands": ["Nike", "Adidas", "Puma", "Asics", "New Balance", "Skechers", "Reebok", "Under Armour",
                   "Timberland", "Clarks", "Crocs", "Birkenstock", "Vans", "Converse", "Woodland"],
        "items": ["Air Max Sneakers", "Ultraboost Running Shoes", "Nitro Trail Runners", "Gel-Kayano Stability Shoes",
                  "Classic Lifestyle Sneakers", "Walking Comfort Shoes", "Retro Court Shoes", "Waterproof Hiking Boots",
                  "Leather Chukka Boots", "Classic Clogs", "Two-Strap Sandals", "Canvas High-Tops",
                  "Skate Leather Shoes", "Slip-On Loafers", "Athletic Training Shoes"],
        "min_p": 1200, "max_p": 22000, "rating_loc": 4.4, "rating_scale": 0.25,
        "images": [
            "photo-1542291026-7eec264c27ff", # Air Max Sneakers
            "photo-1606107557195-0e29a4b5b4aa", # Ultraboost Running Shoes
            "photo-1608231387042-66d1773070a5", # Nitro Trail Runners
            "photo-1539185441755-769473a23570", # Gel-Kayano Stability Shoes
            "photo-1595950653106-6c9ebd614d3a", # Classic Lifestyle Sneakers
            "photo-1549298916-b41d501d3772", # Walking Comfort Shoes
            "photo-1607522370275-f14206abe5d3", # Retro Court Shoes
            "photo-1560343090-f0409e92791a", # Waterproof Hiking Boots
            "photo-1549298916-b41d501d3772", # Leather Chukka Boots
            "photo-1520639888713-7851133b1ed0", # Classic Clogs
            "photo-1603808033192-082d6f74b30e", # Two-Strap Sandals
            "photo-1607522370275-f14206abe5d3", # Canvas High-Tops
            "photo-1618677831708-0e7fda3148b4", # Skate Leather Shoes
            "photo-1533867617858-e7b97e060509", # Slip-On Loafers
            "photo-1606107557195-0e29a4b5b4aa"  # Athletic Training Shoes
        ],
        "desc": "Engineered for ultimate comfort and durability, the {brand} {item} ({modifier}) features responsive cushioning, slip-resistant soles, and a sleek modern silhouette perfect for daily wear."
    },
    "clothing": {
        "brands": ["Levi's", "Patagonia", "The North Face", "Nike", "Adidas", "Uniqlo", "Ralph Lauren",
                   "Tommy Hilfiger", "Calvin Klein", "Columbia", "Carhartt", "Zara", "H&M", "Arc'teryx", "Under Armour"],
        "items": ["Slim Fit Jeans", "Torrentshell Rain Jacket", "Nuptse Down Parka", "Fleece Pullover Hoodie",
                  "Tiro Track Pants", "AIRism Cotton Tee", "Pique Cotton Polo", "Oxford Button-Down Shirt",
                  "Classic Chino Pants", "Full-Zip Fleece Jacket", "Rugged Work Jacket", "Cargo Utility Pants",
                  "Merino Wool Sweater", "Windbreaker Jacket", "Thermal Base Layer"],
        "min_p": 700, "max_p": 35000, "rating_loc": 4.3, "rating_scale": 0.28,
        "images": [
            "photo-1541099649105-f69ad21f3246", # Slim Fit Jeans
            "photo-1544441893-675973e31985", # Torrentshell Rain Jacket
            "photo-1591047139829-d91aecb6caea", # Nuptse Down Parka
            "photo-1556821840-3a63f95609a7", # Fleece Pullover Hoodie
            "photo-1551799517-eb8f03cb5e6a", # Tiro Track Pants
            "photo-1521572267360-ee0c2909d518", # AIRism Cotton Tee
            "photo-1581655353564-df123a1eb820", # Pique Cotton Polo
            "photo-1596755094514-f87e34085b2c", # Oxford Button-Down Shirt
            "photo-1624378439575-d8705ad7ae80", # Classic Chino Pants
            "photo-1507679799987-c73779587ccf", # Full-Zip Fleece Jacket
            "photo-1551028719-00167b16eac5", # Rugged Work Jacket
            "photo-1624378439575-d8705ad7ae80", # Cargo Utility Pants
            "photo-1620799140408-edc6dcb6d633", # Merino Wool Sweater
            "photo-1548883354-7622d03aca27", # Windbreaker Jacket
            "photo-1602810318383-e386cc2a3ccf"  # Thermal Base Layer
        ],
        "desc": "Tailored from exceptionally soft, breathable materials, this {brand} {item} ({modifier}) provides a sophisticated look and all-day comfort. An essential addition to any curated wardrobe."
    },
    "beauty": {
        "brands": ["COSRX", "The Ordinary", "CeraVe", "La Roche-Posay", "Laneige", "Paula's Choice",
                   "Estée Lauder", "Clinique", "Kiehl's", "Glow Recipe", "Dior", "Chanel", "YSL", "Olaplex", "Mac"],
        "items": ["Snail Mucin Essence", "Niacinamide Serum", "Moisturizing Cream", "SPF 50+ Sunscreen",
                  "Lip Sleeping Mask", "2% BHA Liquid Exfoliant", "Advanced Night Repair", "Hydrating Auto-Replenisher",
                  "Facial Fuel Cleanser", "Watermelon Glow Toner", "Sauvage Eau de Parfum",
                  "Bleu de Chanel Intense", "Radiant Creamy Concealer", "Matte Bullet Lipstick", "Hair Perfector Bond Repair"],
        "min_p": 400, "max_p": 15000, "rating_loc": 4.5, "rating_scale": 0.22,
        "images": [
            "photo-1608248597481-496100c80836", # Snail Mucin
            "photo-1571781926291-c477ebfd024b", # Niacinamide
            "photo-1556228720-195a672e8a03", # Cream
            "photo-1598440947619-2c35fc9aa908", # Sunscreen
            "photo-1620916566398-39f1143ab7be", # Lip Mask
            "photo-1612817288484-6f916006741a", # Liquid Exfoliant
            "photo-1571781926291-c477ebfd024b", # Night Repair
            "photo-1620916566398-39f1143ab7be", # Auto-Replenisher
            "photo-1556228720-195a672e8a03", # Cleanser
            "photo-1608248597481-496100c80836", # Toner
            "photo-1526947425960-945c6e72858f", # Sauvage
            "photo-1526947425960-945c6e72858f", # Bleu de Chanel
            "photo-1620916566398-39f1143ab7be", # Concealer
            "photo-1617897903246-719242758050", # Lipstick
            "photo-1608248597481-496100c80836"  # Bond Repair
        ],
        "desc": "Formulated with dermatologist-approved active ingredients, this {brand} {item} ({modifier}) targets skin vitality, offering a radiant, hydrated complexion and visible results."
    },
    "home_decor": {
        "brands": ["Philips Hue", "IKEA", "Bath & Body Works", "Marshall", "Safavieh", "Umbra", "Zinus",
                   "Casper", "West Elm", "Home Centre", "Pottery Barn", "Dyson", "Mueller", "Target", "Wayfair"],
        "items": ["Smart LED Starter Kit", "Minimalist Wall Clock", "Scented Triple-Wick Candle",
                  "Abstract Canvas Wall Art", "Woven Cotton Table Runner", "Ceramic Flower Vase Set",
                  "Faux Fur Throw Blanket", "Magnetic Moon Desk Lamp", "Floating Wooden Shelves",
                  "Himalayan Pink Salt Lamp", "Velvet Accent Armchair", "Blended Area Rug",
                  "Geometric Desktop Planter", "Studio Writing Desk", "Pure Cool Purifying Fan"],
        "min_p": 500, "max_p": 45000, "rating_loc": 4.3, "rating_scale": 0.28,
        "images": [
            "photo-1550985543-f47f38aeee65", # Smart LED
            "photo-1563861826100-9cb868fdcd1d", # Wall Clock
            "photo-1603006905003-be475563bc59", # Candle
            "photo-1513519245088-0e12902e5a38", # Abstract Canvas
            "photo-1513519245088-0e12902e5a38", # Table Runner
            "photo-1578500494198-246f612d3b3d", # Ceramic Vase
            "photo-1540518614846-7eded433c457", # Throw Blanket
            "photo-1507473885765-e6ed057f782c", # Desk Lamp
            "photo-1595428774223-ef52624120d2", # Floating Shelves
            "photo-1507473885765-e6ed057f782c", # Salt Lamp
            "photo-1598300042247-d088f8ab3a91", # Velvet Armchair
            "photo-1600121848594-d8644e57abab", # Area Rug
            "photo-1485955900006-10f4d324d411", # Desktop Planter
            "photo-1518455027359-f3f8164ba6bd", # Writing Desk
            "photo-1585338107529-13afc5f02586"  # Purifying Fan
        ],
        "desc": "Elevate your living space with this artisan-crafted {brand} {item} ({modifier}). Blends timeless aesthetics with modern functionality, creating a warm and sophisticated ambiance."
    },
    "fitness": {
        "brands": ["Bowflex", "Manduka", "TRX", "Optimum Nutrition", "Myprotein", "Theragun", "Fitbit",
                   "Garmin", "Rogue", "Peloton", "Concept2", "Hydro Flask", "BlenderBottle", "Gymshark", "Under Armour"],
        "items": ["SelectTech Adjustable Dumbbells", "Pro Yoga Mat 6mm", "Suspension Training Kit",
                  "Gold Standard Whey 2kg", "Impact Whey Isolate", "Deep Tissue Massager Gun",
                  "Smart Health Tracker", "GPS Running Smartwatch", "Olympic Steel Barbell",
                  "Indoor Stationary Bike", "Ergonomic Rowing Machine", "Wide Mouth Water Bottle",
                  "Classic Shaker Cup", "Zip-Up Track Jacket", "Compression Training Shorts"],
        "min_p": 300, "max_p": 180000, "rating_loc": 4.4, "rating_scale": 0.25,
        "images": [
            "photo-1638536532686-d610adfc8e5c", # Dumbbells
            "photo-1592432678016-e910b452f9a2", # Yoga Mat
            "photo-1517838277536-f5f99be501cd", # TRX Kit
            "photo-1579758629938-03607ccdbaba", # Whey
            "photo-1579758629938-03607ccdbaba", # Whey Isolate
            "photo-1548690312-e3b507d8c110", # Massager Gun
            "photo-1508685096489-7aacd43bd3b1", # Smart Tracker
            "photo-1508685096489-7aacd43bd3b1", # GPS Smartwatch
            "photo-1517838277536-f5f99be501cd", # Steel Barbell
            "photo-1594737626072-90dc274bc2bd", # Indoor Bike
            "photo-1594737626072-90dc274bc2bd", # Rowing Machine
            "photo-1602143407151-7111542de6e8", # Water Bottle
            "photo-1579758629938-03607ccdbaba", # Shaker Cup
            "photo-1548690312-e3b507d8c110", # Track Jacket
            "photo-1548690312-e3b507d8c110"  # Compression Shorts
        ],
        "desc": "Achieve your peak performance with the {brand} {item} ({modifier}). Built from heavy-duty, gym-grade materials, it's designed to withstand intense workouts and optimize your training metrics."
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
        "min_p": 250, "max_p": 8000, "rating_loc": 4.5, "rating_scale": 0.20,
        "images": [
            "photo-1544947950-fa07a98d237f", # Designing Data
            "photo-1506880018603-83d5b814b5a6", # Deep Learning
            "photo-1544947950-fa07a98d237f", # Atomic Habits
            "photo-1495446815901-a7297e633e8d", # Lean Startup
            "photo-1512820790803-83ca734da794", # Zero to One
            "photo-1516979187457-637abb4f9353", # Clean Code
            "photo-1532012197267-da84d127e765", # Algorithms
            "photo-1544947950-fa07a98d237f", # Intelligent Investor
            "photo-1506880018603-83d5b814b5a6", # Thinking Fast
            "photo-1512820790803-83ca734da794", # Sapiens
            "photo-1495446815901-a7297e633e8d", # Principles
            "photo-1516979187457-637abb4f9353", # Psychology of Money
            "photo-1589829085413-56de8ae18c73", # Continuous Delivery
            "photo-1532012197267-da84d127e765", # Pragmatic Programmer
            "photo-1544947950-fa07a98d237f"  # Grokking Algorithms
        ],
        "desc": "A masterfully written guide, this edition of {brand} {item} ({modifier}) delivers expert strategies, deep industry insights, and actionable advice to accelerate your learning journey."
    },
    "automotive": {
        "brands": ["Vantrue", "70mai", "Anker Roav", "Chemical Guys", "Meguiar's", "NOCO", "Baseus",
                   "Michelin", "Philips", "Bosch", "Spigen", "Garmin", "Armor All", "ThisWorx", "WeatherTech"],
        "items": ["3-Channel Dash Cam", "Dual Lens Smart Dashcam", "SmartCharge FM Transmitter",
                  "Complete Car Wash Kit", "Premium Liquid Wax", "1000A Battery Jump Starter",
                  "Portable Tyre Inflator", "Digital Pressure Gauge", "X-tremeVision Headlight Bulbs",
                  "ClearMax Wiper Blades", "Magnetic Car Phone Mount", "GPS Smart Navigator",
                  "Interior Glass Wipes Tube", "High Power Car Vacuum", "Laser Measured Floor Mats"],
        "min_p": 300, "max_p": 35000, "rating_loc": 4.3, "rating_scale": 0.28,
        "images": [
            "photo-1563720223185-11003d516935", # Dash Cam
            "photo-1563720223185-11003d516935", # Smart Dashcam
            "photo-1549399542-7e3f8b79c341", # FM Transmitter
            "photo-1607860108855-64acf2078ed9", # Car Wash Kit
            "photo-1607860108855-64acf2078ed9", # Liquid Wax
            "photo-1619642751034-765dfdf7c58e", # Jump Starter
            "photo-1619642751034-765dfdf7c58e", # Tyre Inflator
            "photo-1619642751034-765dfdf7c58e", # Pressure Gauge
            "photo-1486006920555-c77dce18193b", # Bulbs
            "photo-1563720223185-11003d516935", # Wiper Blades
            "photo-1586444248902-2f64eddc13df", # Phone Mount
            "photo-1617788138017-80ad40651399", # GPS Navigator
            "photo-1607860108855-64acf2078ed9", # Glass Wipes
            "photo-1563720223185-11003d516935", # Car Vacuum
            "photo-1549399542-7e3f8b79c341"  # Floor Mats
        ],
        "desc": "Protect and optimize your vehicle with the high-performance {brand} {item} ({modifier}). Easy to install and constructed from rugged materials, it ensures longevity and road safety."
    },
    "toys": {
        "brands": ["LEGO", "Hasbro", "Mattel", "Rubik's", "DJI Ryze", "Nerf", "Hot Wheels", "Barbie",
                   "Funko", "Fisher-Price", "Ravensburger", "Melissa & Doug", "Play-Doh", "Sphero", "Exploding Kittens"],
        "items": ["Star Wars Building Kit", "Technic Supercar Set", "Catan Strategy Board Game",
                  "Monopoly Ultimate Banking", "Uno Flip Card Edition", "Bluetooth 3x3 Smart Cube",
                  "Tello Mini Drone", "Commander Rotating Blaster", "20-Car Gift Pack Assortment",
                  "Dreamhouse Playset", "Marvel Collectible Figure", "Smart Stages Interactive Puppy",
                  "Villainous Strategy Expansion", "Wooden Engineering Blocks", "Modeling Compound 24-Pack"],
        "min_p": 150, "max_p": 85000, "rating_loc": 4.4, "rating_scale": 0.26,
        "images": [
            "photo-1587654780291-39c9404d746b", # Building Kit
            "photo-1566577134770-3d85bb3a9cc4", # Supercar Set
            "photo-1610890716171-6b1bb98ffd09", # Catan Game
            "photo-1610890716171-6b1bb88ffd09", # Monopoly
            "photo-1610890716171-6b1bb98ffd09", # Uno Flip
            "photo-1591989330344-77a83d73010b", # Smart Cube
            "photo-1527977966376-1c8408f9f108", # Mini Drone
            "photo-15958060370-d644479cb6f7", # Blaster
            "photo-1594787318286-3d835c1d207f", # Gift Pack Assortment
            "photo-1515488042361-404e9250afef", # Dreamhouse Playset
            "photo-1608889175123-8ec330b86f84", # Collectible Figure
            "photo-1596461404969-9ae70f2830c1", # Interactive Puppy
            "photo-1610890716171-6b1bb98ffd09", # Villainous Expansion
            "photo-1618842676088-c4d48a6a7c9d", # Wooden Blocks
            "photo-1596461404969-9ae70f2830c1"  # Modeling Compound
        ],
        "desc": "Ignite imagination and cognitive development with the {brand} {item} ({modifier}). Featuring interactive components and high-quality build materials, it provides hours of engaging fun."
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
        "min_p": 50, "max_p": 4500, "rating_loc": 4.4, "rating_scale": 0.22,
        "images": [
            "photo-1497515114629-f71d768fd07c", # Coffee Beans
            "photo-1576092768241-dec231879fc3", # Tea Bags
            "photo-1474979266404-7eaacbcd87c5", # Olive Oil
            "photo-1508061253366-f7da158b6cd9", # Almonds
            "photo-1590080875515-8a3a8dc5735e", # Peanut Butter
            "photo-1606313564200-e75d5e30476c", # Dark Chocolate
            "photo-1586444248902-2f64eddc13df", # Rolled Oats
            "photo-1587049352846-4a222e784d38", # Himalayan Honey
            "photo-158444248902-2f64eddc13df", # Chia Seeds
            "photo-1586201375761-83865001e31c", # Basmati Rice
            "photo-1586444248902-2f64eddc13df", # Corn Flakes
            "photo-1474979266404-7eaacbcd87c5", # Cooking Oil
            "photo-1606313564200-e75d5e30476c", # Cocoa
            "photo-1578985545062-69928b1d9587", # Biscuit Spread
            "photo-1578985545062-69928b1d9587"  # Cocoa Spread
        ],
        "desc": "Sourced from the finest natural ingredients, this organic {brand} {item} ({modifier}) offers rich, authentic flavors and dense nutritional value. Perfect for gourmet culinary experiences."
    },
}

MODIFIERS = [
    "Pro", "Max", "Ultra", "Elite", "Classic", "Series II", "v2",
    "Edition", "Premium", "Advanced", "Signature", "Select", "Core", "Lite", "Plus",
]

PRODUCTS_PER_CATEGORY = 200

REVIEWERS = [
    "Alex M.", "S. Taylor", "Jordan K.", "Emma Watson", "Rohan Gupta", "Clara R.",
    "Priyesh S.", "David Miller", "Elena Rostova", "Hiroshi Tanaka", "Sarah Jenkins", "Michael O'Connor"
]

CATEGORY_REVIEWS = {
    "electronics": {
        "pos": ["Absolutely amazing quality! The display/sound is top-notch.",
                "Well worth the premium price. Sleek design and works perfectly.",
                "Very impressed with the build quality. Apple-like precision.",
                "Exceeded my expectations. The speed and screen quality are unmatched."],
        "neg": ["Decent, but the battery life is slightly lower than expected.",
                "A bit overpriced for the specs, although the finish is nice.",
                "Had some connectivity issues at first, but it works okay now.",
                "It's good, but the user manual was missing."]
    },
    "footwear": {
        "pos": ["Incredibly comfortable! Feels like walking on clouds.",
                "Perfect fit and great arch support. Highly recommend.",
                "Very stylish and the materials feel highly durable.",
                "Amazing traction and look. Got compliments on day one."],
        "neg": ["A bit tight around the toes. I suggest ordering half a size up.",
                "Comfortable but the color is slightly different from the photos.",
                "The laces are a bit short, but otherwise a solid pair of shoes."]
    },
    "clothing": {
        "pos": ["Stunning fabric and fit! Drapes perfectly on the shoulders.",
                "Excellent stitching. Breathable, premium texture.",
                "Feels very high quality. Fits true to size and handles washing well.",
                "Incredibly cozy and stylish. A staple in my collection now."],
        "neg": ["Shrank a little bit after the first wash, recommend cold water only.",
                "Sleeve length is slightly longer than standard sizes.",
                "Nice texture, but the color is slightly duller than online photos."]
    },
    "beauty": {
        "pos": ["Immediate difference in skin texture! So soft and glowing.",
                "Amazing scent and non-greasy formula. Extremely hydrating.",
                "A holy grail product. Cleared my skin barrier in a week.",
                "Exquisite packaging and luxurious application feel."],
        "neg": ["Works well but has a slightly strong medicinal scent.",
                "Takes some time to absorb fully. Best for night use.",
                "A bit heavy for very oily skin types, but good quality overall."]
    },
    "home_decor": {
        "pos": ["Beautiful craftsmanship. Added an instant touch of class to my living room.",
                "Stunning colors and robust materials. Very premium build.",
                "Exceeded my design expectations. Looks exactly like high-end boutique decor.",
                "Extremely aesthetic! Brings the whole room design together."],
        "neg": ["A bit smaller than I envisioned, but still looks lovely.",
                "The lighting/shade is slightly warmer than shown in illustrations.",
                "Instruction sheet for wall mounting was slightly confusing."]
    },
    "fitness": {
        "pos": ["Extremely durable build. Survives intensive daily training sessions.",
                "Ergonomic design makes workouts significantly smoother.",
                "Top-tier performance metrics. Highly recommend for enthusiasts.",
                "Exceptional build and stability. Commercial gym quality at home."],
        "neg": ["Quite heavy to move around, but highly stable once in place.",
                "A bit expensive compared to entry-level alternatives.",
                "Setup took about an hour, although instructions were clear."]
    },
    "books": {
        "pos": ["A absolute masterpiece. Actionable ideas and exceptionally written.",
                "Stunning illustrations and page quality. Must-have on your bookshelf.",
                "Deep, thought-provoking insights that shifted my perspective entirely.",
                "Clear explanations of complex concepts. Invaluable reference book."],
        "neg": ["Very dense reading, takes some time to fully digest the material.",
                "Cover had a small crease upon arrival, but content is gold.",
                "A bit academic in tone, could have included more real-world examples."]
    },
    "automotive": {
        "pos": ["Robust, high-grade utility. Fits like a glove in my car dashboard.",
                "Extremely easy to setup and provides outstanding results.",
                "Heavy-duty construction. Resists wear and weather beautifully.",
                "Exceptional performance. Solved my vehicle issue instantly."],
        "neg": ["Cable length was slightly shorter than needed for larger SUVs.",
                "App interface is a bit basic, though the hardware is great.",
                "Instructions could have been slightly more detailed."]
    },
    "toys": {
        "pos": ["Keeps the kids engaged for hours! Fantastic design and logic.",
                "Exceptional durability. Safe materials and bright, vivid colors.",
                "Very creative design. Stimulates problem solving effectively.",
                "Outstanding build and precision engineering. A joy to assemble."],
        "neg": ["Lots of tiny pieces, make sure you don't lose them!",
                "Required adult assistance to build, but a great bonding activity.",
                "Battery compartment was slightly tricky to open."]
    },
    "groceries": {
        "pos": ["Incredible depth of flavor. Fresh, aromatic, and rich quality.",
                "Pure, natural taste with outstanding packaging to preserve freshness.",
                "Top-tier organic quality. You can really taste the difference.",
                "Rich, velvety texture. One of the best brands I've tried."],
        "neg": ["A bit pricey for the quantity, but taste is premium.",
                "Short expiration window, since it doesn't contain artificial preservatives.",
                "Jar lid was sealed very tight, needed some effort to open."]
    }
}


def _generate_products_and_reviews(rng: np.random.Generator) -> tuple[list, list]:
    product_rows = []
    review_rows = []
    product_id_counter = 1

    for category in CATEGORIES:
        mx = SEED_MATRICES[category]
        brands = mx["brands"]
        items = mx["items"]
        min_p, max_p = mx["min_p"], mx["max_p"]
        rating_loc, rating_scale = mx["rating_loc"], mx["rating_scale"]
        images = mx["images"]
        desc_tmpl = mx["desc"]

        seen_names = set()
        generated = 0

        while generated < PRODUCTS_PER_CATEGORY:
            brand = rng.choice(brands)
            item = rng.choice(items)
            modifier = rng.choice(MODIFIERS)

            name = f"{brand} {item} ({modifier})"

            if name in seen_names:
                continue

            seen_names.add(name)

            price = int(rng.uniform(min_p, max_p))
            rating = float(np.clip(rng.normal(rating_loc, rating_scale), 1.0, 5.0))
            rating = round(rating, 1)

            stock = int(rng.choice([0, 3, 5, 12, 25, 45, 80, 150]))
            description = desc_tmpl.format(brand=brand, item=item, modifier=modifier)

            # Map to a deterministic image url based strictly on item type
            item_idx = items.index(item)
            img_id = images[item_idx]
            image_url = f"https://images.unsplash.com/{img_id}?auto=format&fit=crop&w=600&q=80"

            # Create product row
            product_rows.append((
                product_id_counter, name, category, price, rating,
                brand, item, modifier, stock, description, image_url
            ))

            # Create 3 reviews for this product
            review_templates = CATEGORY_REVIEWS.get(category, CATEGORY_REVIEWS["electronics"])
            for r_idx in range(3):
                reviewer = rng.choice(REVIEWERS)
                # Pick rating stars based on overall rating
                if rating >= 4.0:
                    stars = int(rng.choice([4, 5]))
                    comment = rng.choice(review_templates["pos"])
                else:
                    stars = int(rng.choice([2, 3, 4]))
                    comment = rng.choice(review_templates["neg"] + review_templates["pos"])

                review_rows.append((
                    product_id_counter, reviewer, stars, comment
                ))

            product_id_counter += 1
            generated += 1

        print(f"  ✓  {category:<12}  {generated} products (with {generated * 3} reviews)")

    return product_rows, review_rows


def seed_complete_database() -> None:
    rng = np.random.default_rng(seed=101)

    print("Generating products and reviews …")
    products, reviews = _generate_products_and_reviews(rng)

    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    cur.executescript("""
        DROP TABLE IF EXISTS reviews;
        DROP TABLE IF EXISTS products;
        
        CREATE TABLE products (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL,
            category    TEXT    NOT NULL,
            price       REAL    NOT NULL,
            rating      REAL    NOT NULL,
            brand       TEXT,
            item        TEXT,
            modifier    TEXT,
            stock       INTEGER,
            description TEXT,
            image_url   TEXT
        );
        
        CREATE TABLE reviews (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id  INTEGER NOT NULL,
            user        TEXT    NOT NULL,
            stars       INTEGER NOT NULL,
            comment     TEXT    NOT NULL,
            timestamp   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
        );
        
        CREATE INDEX IF NOT EXISTS idx_cat_price ON products (category, price);
        CREATE INDEX IF NOT EXISTS idx_prod_reviews ON reviews (product_id);
    """)

    cur.executemany(
        """INSERT INTO products 
           (id, name, category, price, rating, brand, item, modifier, stock, description, image_url) 
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        products,
    )

    cur.executemany(
        "INSERT INTO reviews (product_id, user, stars, comment) VALUES (?, ?, ?, ?)",
        reviews,
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

    cur.execute("SELECT COUNT(*) FROM reviews")
    total_reviews = cur.fetchone()[0]
    print(f"  {'REVIEWS':<14} {total_reviews:>4}")

    conn.close()
    print(f"\ninventory.db ready — {total} products and {total_reviews} reviews seeded.")


if __name__ == "__main__":
    seed_complete_database()
