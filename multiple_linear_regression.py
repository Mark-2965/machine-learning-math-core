import numpy as np

print("🤖 INITIALIZING BISMARK DATA LABS MULTI-VARIABLE ML CORE...\n")

# 📊 1. PREPARE THE MULTI-VARIABLE DATASET (Features Matrix & Target Vector)
# Columns: [Square Footage, Number of Bedrooms]
X_data = np.array([[1000, 2], [1500, 3], [2000, 3], [3000, 5]])

# Corresponding actual prices in Millions of Naira (₦)
y_prices = np.array([30.0, 45.0, 55.0, 95.0])

# 🧠 2. ADVANCED MATRIX INTERCEPT PLUMBING (Adding a bias column of 1s)
# This lets the model calculate a base intercept price automatically
bias_column = np.ones((X_data.shape[0], 1))
X_matrix = np.hstack((bias_column, X_data))

# 🏋️ 3. RUNNING MATRIX LEAST SQUARES CALCULUS: (X^T * X)^(-1) * X^T * y
X_transpose = X_matrix.T
coefficients = np.linalg.inv(X_transpose @ X_matrix) @ X_transpose @ y_prices

# Extracting the calculated pricing rules
base_intercept = coefficients[0]
price_per_sqft = coefficients[1]
price_per_bedroom = coefficients[2]

print("✅ MULTI-VARIABLE PATTERN EXTRACTION LOCKED IN.")
print(f"• Base Intercept Price: ₦{base_intercept * 1000000:,.2f}")
print(f"• Size Coefficient:    ₦{price_per_sqft * 1000000:,.2f} per sq ft")
print(f"• Room Coefficient:    ₦{price_per_bedroom * 1000000:,.2f} per bedroom")
print("-" * 60)

# 🔮 4. PREDICT A BRAND NEW, COMPLEX LEKKI HOUSE TARGET
target_size = 2500
target_bedrooms = 4

# Run matrix dot product inference calculation
new_house_features = np.array([1, target_size, target_bedrooms])
predicted_price = new_house_features @ coefficients

print(f"🔮 [MULTI-VARIABLE PREDICTION RESULT]:")
print(f"For a brand new {target_size} sq ft house with {target_bedrooms} bedrooms in Lekki:")
print(f"➔ AI Estimated Value: ₦{predicted_price:.2f} Million Naira!")
print("-" * 60)
