import numpy as np

print("🤖 INITIALIZING BISMARK DATA LABS ML MATH CORE ENGINES...\n")

# 📊 1. PREPARE THE TRAINING DATA (Lekki / Ajah Case Study)
# House sizes in square feet
x = np.array([1000, 1500, 2000, 3000]) 
# Corresponding actual prices in Millions of Naira (₦)
y = np.array([30, 45, 60, 90]) 

# 🧠 2. PURE ML CALCULUS: RUNNING NUMERICAL SLOPE EXTRACTION
# Calculating the exact linear trendline factor: Price = Weight * Size
weight = np.sum(x * y) / np.sum(x * x)

print("✅ MATHEMATICAL RULE PATTERN EXTRACTION LOCKED IN.")
print(f"• Extracted Coefficient: ₦{weight * 1000000:,.2f} per sq ft")
print("-" * 60)

# 🔮 3. PREDICT THE PRICE OF A BRAND NEW HOUSE IN LEKKI!
target_house_size = 2500 # A house size we never showed the model!
predicted_price = weight * target_house_size

print(f"🔮 [PREDICTION RESULT]:")
print(f"For a brand new {target_house_size} sq ft house in Lekki, the model calculates the price at:")
print(f"➔ ₦{predicted_price:.2f} Million Naira!")
print("-" * 60)
