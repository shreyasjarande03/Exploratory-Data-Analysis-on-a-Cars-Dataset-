"""
Exploratory Data Analysis (EDA) on Cars Dataset
Author: Student Project Assignment
Dataset: CarDekho Used Car Dataset (Car details v3.csv)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure figures directory exists
os.makedirs("figures", exist_ok=True)

# Set global aesthetic theme
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica', 'Arial', 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

print("=" * 70)
print("STEP 1: DATA LOADING & INITIAL INSPECTION")
print("=" * 70)

# Load dataset
df_raw = pd.read_csv("Car_details_v3.csv")
print(f"Raw Dataset Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
print("\nRaw Columns and Data Types:")
print(df_raw.dtypes)

print("\nMissing Values Count:")
print(df_raw.isnull().sum())

duplicate_count_raw = df_raw.duplicated().sum()
print(f"\nExact Duplicate Rows Detected: {duplicate_count_raw} ({duplicate_count_raw / len(df_raw) * 100:.2f}%)")

print("\n" + "=" * 70)
print("STEP 2: DATA CLEANING & FEATURE RESTRUCTURING")
print("=" * 70)

# 1. Remove exact duplicate rows
df = df_raw.drop_duplicates().copy()
print(f"Dataset Size after removing duplicates: {df.shape[0]} rows")

# 2. Extract brand from vehicle name
df['brand'] = df['name'].apply(lambda x: str(x).split()[0].strip())
print(f"Extracted {df['brand'].nunique()} unique automotive brands.")

# 3. Clean 'mileage': extract numeric value and convert 0.0 to NaN (implausible value)
def parse_mileage(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    try:
        num = float(val_str.split()[0])
        return np.nan if num <= 0 else num
    except:
        return np.nan

df['mileage_kmpl'] = df['mileage'].apply(parse_mileage)

# 4. Clean 'engine': extract numeric CC displacement
def parse_engine(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).replace('CC', '').strip()
    try:
        return float(val_str)
    except:
        return np.nan

df['engine_cc'] = df['engine'].apply(parse_engine)

# 5. Clean 'max_power': extract numeric BHP, handling empty strings like ' bhp'
def parse_power(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).replace('bhp', '').strip()
    try:
        return float(val_str)
    except:
        return np.nan

df['max_power_bhp'] = df['max_power'].apply(parse_power)

# 6. Clean seats
df['seats'] = pd.to_numeric(df['seats'], errors='coerce')

# 7. Calculate vehicle age (listing anchor year = 2020)
anchor_year = 2020
df['vehicle_age'] = anchor_year - df['year']
df['vehicle_age'] = df['vehicle_age'].clip(lower=0)

# 8. Create human-interpretable Price in Lakhs (INR 1 Lakh = 100,000) & Log Price
df['price_lakh'] = df['selling_price'] / 100000.0
df['log_price'] = np.log1p(df['selling_price'])
df['log_km_driven'] = np.log1p(df['km_driven'])

# 9. Handle missing technical parameters
tech_missing = df[['mileage_kmpl', 'engine_cc', 'max_power_bhp', 'seats']].isnull().any(axis=1)
print(f"Records with missing or invalid technical specifications: {tech_missing.sum()}")

df_clean = df.dropna(subset=['mileage_kmpl', 'engine_cc', 'max_power_bhp', 'seats']).copy()
df_clean['seats'] = df_clean['seats'].astype(int)
df_clean['engine_cc'] = df_clean['engine_cc'].astype(int)

# Standardize categorical strings
df_clean['fuel'] = df_clean['fuel'].str.strip()
df_clean['seller_type'] = df_clean['seller_type'].str.strip()
df_clean['transmission'] = df_clean['transmission'].str.strip()
df_clean['owner'] = df_clean['owner'].str.strip()

print(f"\nFinal Cleaned Dataset Dimensions: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
print(f"Total rows filtered out: {len(df_raw) - len(df_clean)} ({(len(df_raw) - len(df_clean))/len(df_raw)*100:.2f}%)")

print("\n" + "=" * 70)
print("STEP 3: DESCRIPTIVE STATISTICAL ANALYSIS")
print("=" * 70)

num_cols = ['selling_price', 'price_lakh', 'vehicle_age', 'km_driven', 'mileage_kmpl', 'engine_cc', 'max_power_bhp', 'seats']
desc_stats = df_clean[num_cols].describe().T
desc_stats['median'] = df_clean[num_cols].median()
desc_stats['skew'] = df_clean[num_cols].skew()
desc_stats['kurtosis'] = df_clean[num_cols].kurtosis()
cols_order = ['count', 'mean', 'std', 'min', '25%', 'median', '75%', 'max', 'skew', 'kurtosis']
desc_stats = desc_stats[cols_order]
print("Numerical Descriptive Summary:")
print(desc_stats.round(2).to_string())

cat_cols = ['fuel', 'seller_type', 'transmission', 'owner']
print("\nCategorical Distributions:")
for col in cat_cols:
    counts = df_clean[col].value_counts()
    pcts = df_clean[col].value_counts(normalize=True) * 100
    cat_summary = pd.DataFrame({'Count': counts, 'Percentage (%)': pcts.round(2)})
    print(f"\n--- {col.upper()} ---")
    print(cat_summary.to_string())

# Save cleaned dataset
df_clean.to_csv("Car_details_v3_cleaned.csv", index=False)
print("\nCleaned dataset saved to 'Car_details_v3_cleaned.csv'")

print("\n" + "=" * 70)
print("STEP 4: GENERATING HIGH-RESOLUTION CHARTS")
print("=" * 70)

# ----------------------------------------------------
# CHART 1: Price Distribution (Raw Price in Lakhs vs Log Price)
# Type: Histogram & KDE
# ----------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

# Subplot 1: Right-skewed distribution in Lakhs
sns.histplot(df_clean['price_lakh'], bins=40, kde=True, ax=axes[0], color='#1f77b4', edgecolor='white', alpha=0.7)
axes[0].set_title("Distribution of Car Selling Price (INR Lakhs)", fontsize=13, fontweight='bold', pad=12)
axes[0].set_xlabel("Selling Price (INR in Lakhs)", fontsize=11, fontweight='bold')
axes[0].set_ylabel("Vehicle Count", fontsize=11, fontweight='bold')
axes[0].set_xlim(0, 40)
median_p = df_clean['price_lakh'].median()
mean_p = df_clean['price_lakh'].mean()
axes[0].axvline(median_p, color='#d62728', linestyle='--', linewidth=2, label=f"Median: Rs. {median_p:.2f}L")
axes[0].axvline(mean_p, color='#ff7f0e', linestyle=':', linewidth=2, label=f"Mean: Rs. {mean_p:.2f}L")
axes[0].legend(frameon=True, facecolor='white', framealpha=0.9)

# Subplot 2: Log-transformed price distribution
sns.histplot(df_clean['log_price'], bins=35, kde=True, ax=axes[1], color='#2ca02c', edgecolor='white', alpha=0.7)
axes[1].set_title("Log-Transformed Selling Price Distribution", fontsize=13, fontweight='bold', pad=12)
axes[1].set_xlabel("Natural Log of Selling Price [log(Price)]", fontsize=11, fontweight='bold')
axes[1].set_ylabel("Vehicle Count", fontsize=11, fontweight='bold')
median_log = df_clean['log_price'].median()
axes[1].axvline(median_log, color='#d62728', linestyle='--', linewidth=2, label=f"Median Log: {median_log:.2f}")
axes[1].legend(frameon=True, facecolor='white', framealpha=0.9)

plt.tight_layout()
plt.savefig("figures/fig1_price_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved figures/fig1_price_distribution.png")

# ----------------------------------------------------
# CHART 2: Market Dominance - Top 12 Automobile Brands
# Type: Horizontal Bar / Count Plot
# ----------------------------------------------------
plt.figure(figsize=(10, 6), dpi=300)
top_brands = df_clean['brand'].value_counts().head(12)
palette = sns.color_palette("Blues_r", len(top_brands))
bars = plt.barh(top_brands.index[::-1], top_brands.values[::-1], color=palette, edgecolor='none', height=0.7)

for bar in bars:
    w = bar.get_width()
    pct = (w / len(df_clean)) * 100
    plt.text(w + 35, bar.get_y() + bar.get_height()/2, f"{int(w):,} ({pct:.1f}%)",
             va='center', ha='left', fontsize=9.5, color='#333333', fontweight='semibold')

plt.title("Top 12 Most Prevalent Automotive Brands in Listings", fontsize=14, fontweight='bold', pad=14)
plt.xlabel("Listing Count (Market Frequency)", fontsize=11, fontweight='bold')
plt.ylabel("Car Manufacturer / Brand", fontsize=11, fontweight='bold')
plt.xlim(0, max(top_brands.values) * 1.15)
plt.tight_layout()
plt.savefig("figures/fig2_brand_frequency.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved figures/fig2_brand_frequency.png")

# ----------------------------------------------------
# CHART 3: Selling Price by Fuel Type
# Type: Box Plot & Distribution
# ----------------------------------------------------
plt.figure(figsize=(10, 5.8), dpi=300)
fuel_order = ['Diesel', 'Petrol', 'CNG', 'LPG']
fuel_colors = {'Diesel': '#1f77b4', 'Petrol': '#ff7f0e', 'CNG': '#2ca02c', 'LPG': '#9467bd'}

sns.boxplot(
    data=df_clean, x='fuel', y='price_lakh', hue='fuel', order=fuel_order,
    palette=fuel_colors, width=0.55, showmeans=True, legend=False,
    meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"7"},
    flierprops={"marker":"d", "markersize":3, "alpha":0.35, "markerfacecolor":"#555555"}
)
plt.title("Selling Price Distribution across Fuel Types", fontsize=14, fontweight='bold', pad=14)
plt.xlabel("Engine Fuel Type", fontsize=11, fontweight='bold')
plt.ylabel("Selling Price (INR in Lakhs)", fontsize=11, fontweight='bold')
plt.ylim(0, 35)

for i, fuel in enumerate(fuel_order):
    sub = df_clean[df_clean['fuel'] == fuel]['price_lakh']
    med = sub.median()
    plt.text(i, med + 0.8, f"Rs. {med:.2f}L", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#111111')

plt.tight_layout()
plt.savefig("figures/fig3_price_by_fuel.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved figures/fig3_price_by_fuel.png")

# ----------------------------------------------------
# CHART 4: Vehicle Age vs. Selling Price (by Transmission)
# Type: Scatter Plot with Polynomial Trendline
# ----------------------------------------------------
fig, ax = plt.subplots(figsize=(10.5, 6), dpi=300)
colors_trans = {'Manual': '#3470a3', 'Automatic': '#d95f02'}

sample_df = df_clean.sample(n=min(2500, len(df_clean)), random_state=42)

for trans in ['Manual', 'Automatic']:
    sub = sample_df[sample_df['transmission'] == trans]
    ax.scatter(sub['vehicle_age'], sub['price_lakh'], label=f"{trans} Cars",
               color=colors_trans[trans], alpha=0.45, edgecolors='none', s=26)

for trans in ['Manual', 'Automatic']:
    sub_full = df_clean[df_clean['transmission'] == trans]
    z = np.polyfit(sub_full['vehicle_age'], sub_full['price_lakh'], 2)
    p = np.poly1d(z)
    x_range = np.linspace(sub_full['vehicle_age'].min(), sub_full['vehicle_age'].max(), 100)
    ax.plot(x_range, p(x_range), color=colors_trans[trans], linewidth=3, linestyle='-', label=f"{trans} Depreciation Trend")

ax.set_title("Vehicle Age vs. Selling Price: Depreciation & Transmission Impact", fontsize=14, fontweight='bold', pad=14)
ax.set_xlabel("Vehicle Age (Years Since Manufacture)", fontsize=11, fontweight='bold')
ax.set_ylabel("Selling Price (INR in Lakhs)", fontsize=11, fontweight='bold')
ax.set_ylim(0, 45)
ax.set_xlim(0, 22)
ax.legend(frameon=True, facecolor='white', framealpha=0.95, loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig("figures/fig4_price_vs_age.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved figures/fig4_price_vs_age.png")

# ----------------------------------------------------
# CHART 5: Correlation Heatmap
# Type: Heatmap
# ----------------------------------------------------
plt.figure(figsize=(9, 7.2), dpi=300)
heatmap_cols = ['selling_price', 'price_lakh', 'vehicle_age', 'km_driven', 'mileage_kmpl', 'engine_cc', 'max_power_bhp', 'seats']
corr_matrix = df_clean[heatmap_cols].corr()

display_labels = ['Price (INR)', 'Price (Lakh)', 'Age (Years)', 'Km Driven', 'Mileage (kmpl)', 'Engine (CC)', 'Max Power (bhp)', 'Seats']

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(
    corr_matrix, mask=mask, cmap=cmap, vmin=-1.0, vmax=1.0,
    annot=True, fmt=".2f", square=True, linewidths=1.2,
    cbar_kws={"shrink": 0.82, "label": "Pearson Correlation Coefficient"},
    xticklabels=display_labels, yticklabels=display_labels,
    annot_kws={"size": 10, "weight": "semibold"}
)
plt.title("Correlation Matrix of Automobile Attributes", fontsize=14, fontweight='bold', pad=16)
plt.xticks(rotation=45, ha='right', fontsize=9.5)
plt.yticks(rotation=0, fontsize=9.5)
plt.tight_layout()
plt.savefig("figures/fig5_correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved figures/fig5_correlation_heatmap.png")

# ----------------------------------------------------
# CHART 6: Mean Selling Price across Top 15 Brands
# Type: Bar Plot (Ranked)
# ----------------------------------------------------
plt.figure(figsize=(11, 5.8), dpi=300)
top_15_brands = df_clean['brand'].value_counts().head(15).index
brand_price_mean = df_clean[df_clean['brand'].isin(top_15_brands)].groupby('brand')['price_lakh'].agg(['mean', 'count']).sort_values('mean', ascending=False)

norm_colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(brand_price_mean)))
bars = plt.bar(brand_price_mean.index, brand_price_mean['mean'], color=norm_colors, edgecolor='#444444', linewidth=0.6, width=0.65)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.4, f"Rs. {yval:.1f}L",
             ha='center', va='bottom', fontsize=9, fontweight='bold', color='#222222')

plt.title("Average Selling Price Across Top 15 Automotive Brands (INR Lakhs)", fontsize=14, fontweight='bold', pad=14)
plt.xlabel("Automobile Brand", fontsize=11, fontweight='bold')
plt.ylabel("Mean Selling Price (INR Lakhs)", fontsize=11, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=9.5, fontweight='semibold')
plt.ylim(0, max(brand_price_mean['mean']) * 1.15)
plt.tight_layout()
plt.savefig("figures/fig6_avg_price_by_brand.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved figures/fig6_avg_price_by_brand.png")

# ----------------------------------------------------
# CHART 7: Transmission Type Price Comparison
# Type: Violin Plot & Box Plot
# ----------------------------------------------------
plt.figure(figsize=(8.5, 5.5), dpi=300)
palette_trans = {'Manual': '#4575b4', 'Automatic': '#d73027'}

sns.violinplot(
    data=df_clean, x='transmission', y='price_lakh', hue='transmission', palette=palette_trans,
    cut=0, inner=None, alpha=0.35, legend=False
)
sns.boxplot(
    data=df_clean, x='transmission', y='price_lakh', hue='transmission', palette=palette_trans,
    width=0.25, showmeans=True, legend=False,
    meanprops={"marker":"o", "markerfacecolor":"yellow", "markeredgecolor":"black", "markersize":"8"},
    boxprops=dict(alpha=0.9)
)
plt.title("Selling Price Distribution: Manual vs. Automatic Vehicles", fontsize=14, fontweight='bold', pad=14)
plt.xlabel("Transmission Mechanism", fontsize=11, fontweight='bold')
plt.ylabel("Selling Price (INR in Lakhs)", fontsize=11, fontweight='bold')
plt.ylim(0, 45)

for i, trans in enumerate(['Manual', 'Automatic']):
    sub = df_clean[df_clean['transmission'] == trans]['price_lakh']
    med = sub.median()
    avg = sub.mean()
    plt.text(i, 38, f"Median: Rs. {med:.2f}L\nMean: Rs. {avg:.2f}L\nCount: {len(sub):,}",
             ha='center', va='center', fontsize=9.5, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999999", alpha=0.9))

plt.tight_layout()
plt.savefig("figures/fig7_price_by_transmission.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved figures/fig7_price_by_transmission.png")

# ----------------------------------------------------
# CHART 8: Engine Displacement vs Max Power (colored by Mileage)
# Type: Multivariate Scatter Plot
# ----------------------------------------------------
plt.figure(figsize=(10.5, 6), dpi=300)
sc = plt.scatter(
    df_clean['engine_cc'], df_clean['max_power_bhp'],
    c=df_clean['mileage_kmpl'], cmap='plasma',
    s=28, alpha=0.6, edgecolors='none'
)
cbar = plt.colorbar(sc)
cbar.set_label("Fuel Efficiency / Mileage (kmpl)", fontsize=10, fontweight='bold')

m, b = np.polyfit(df_clean['engine_cc'], df_clean['max_power_bhp'], 1)
x_vals = np.linspace(df_clean['engine_cc'].min(), df_clean['engine_cc'].max(), 100)
plt.plot(x_vals, m*x_vals + b, color='#111111', linestyle='--', linewidth=2, label=f"Trendline: Slope={m:.3f}")

plt.title("Engine Displacement (CC) vs. Maximum Power (BHP) and Mileage", fontsize=14, fontweight='bold', pad=14)
plt.xlabel("Engine Displacement (CC)", fontsize=11, fontweight='bold')
plt.ylabel("Maximum Engine Power (BHP)", fontsize=11, fontweight='bold')
plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left')
plt.tight_layout()
plt.savefig("figures/fig8_mileage_vs_engine_power.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved figures/fig8_mileage_vs_engine_power.png")

print("=" * 70)
print("ALL 8 FIGURES GENERATED SUCCESSFULLY IN figures/ DIRECTORY")
print("=" * 70)
