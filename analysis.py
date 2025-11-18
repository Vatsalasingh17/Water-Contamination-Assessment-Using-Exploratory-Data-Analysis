# ===============================================
# WATER QUALITY EDA – Google Colab Notebook
# ===============================================

# -----------------------------
# 1) Import Libraries
# -----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from google.colab import files

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 4)

# -----------------------------
# 2) Upload CSV
# -----------------------------
print("Upload your water-quality CSV file:")
uploaded = files.upload()

filename = list(uploaded.keys())[0]
df = pd.read_csv(filename)
print("\nLoaded:", filename)

# -----------------------------
# 3) Basic Structure
# -----------------------------
print("\n--- Shape ---")
print(df.shape)

print("\n--- Columns & dtypes ---")
print(df.dtypes)

numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
if "is_safe" in numerical_cols:
    numerical_cols.remove("is_safe")

print("\nNumerical columns:", numerical_cols)
print("Categorical/Target:", "is_safe" if "is_safe" in df.columns else "None")

display(df.head())

# -----------------------------
# 4) Missing Values
# -----------------------------
print("\n--- Missing Values ---")
missing = pd.DataFrame({
    "missing_count": df.isnull().sum(),
    "missing_percent": (df.isnull().mean() * 100).round(2)
})
display(missing)

# Impute numeric missing values (median)
df_clean = df.copy()
for col in numerical_cols:
    if df_clean[col].isnull().sum() > 0:
        med = df_clean[col].median()
        df_clean[col].fillna(med, inplace=True)
        print(f"Filled missing in {col} with median = {med}")

# -----------------------------
# 5) Remove duplicates
# -----------------------------
dups = df_clean.duplicated().sum()
print("\nDuplicate rows:", dups)
if dups > 0:
    df_clean.drop_duplicates(inplace=True)
    print("Duplicates removed.")

# -----------------------------
# 6) Univariate Analysis
# -----------------------------
print("\n--- Univariate Analysis ---")

for col in numerical_cols:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    sns.histplot(df_clean[col], kde=True, ax=ax[0])
    ax[0].set_title(f"Histogram — {col}")

    sns.boxplot(x=df_clean[col], ax=ax[1])
    ax[1].set_title(f"Boxplot — {col}")

    plt.show()

display(df_clean[numerical_cols].describe().T)

# -----------------------------
# 7) Class Distribution (is_safe)
# -----------------------------
if "is_safe" in df_clean.columns:
    print("\n--- Class Distribution ---")
    display(df_clean["is_safe"].value_counts())

    plt.figure(figsize=(6, 4))
    sns.countplot(x=df_clean["is_safe"])
    plt.title("is_safe Class Count")
    plt.show()

# -----------------------------
# 8) Bivariate Analysis (With is_safe)
# -----------------------------
if "is_safe" in df_clean.columns:
    print("\n--- Bivariate Analysis w.r.t is_safe ---")

    for col in numerical_cols:
        plt.figure(figsize=(10, 4))
        sns.boxplot(x="is_safe", y=col, data=df_clean)
        plt.title(f"{col} vs is_safe")
        plt.show()

    print("\nMean values grouped by class:")
    display(df_clean.groupby("is_safe")[numerical_cols].mean())

# -----------------------------
# 9) Correlation Analysis
# -----------------------------
print("\n--- Correlation Matrix ---")
corr = df_clean.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

if "is_safe" in corr.columns:
    print("\nTop positive correlations with is_safe:")
    print(corr["is_safe"].sort_values().tail(5))

    print("\nTop negative correlations with is_safe:")
    print(corr["is_safe"].sort_values().head(5))

# -----------------------------
# 10) Outlier Detection
# -----------------------------
print("\n--- Outlier Detection ---")

iqr_outliers = {}
z_outliers = {}

for col in numerical_cols:
    q1 = df_clean[col].quantile(0.25)
    q3 = df_clean[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    iqr_outliers[col] = (~df_clean[col].between(lower, upper)).sum()

    if df_clean[col].std() > 0:
        z = np.abs(stats.zscore(df_clean[col]))
        z_outliers[col] = (z > 3).sum()
    else:
        z_outliers[col] = 0

outlier_df = pd.DataFrame({"IQR_outliers": iqr_outliers, "Z_outliers": z_outliers})
display(outlier_df)

# -----------------------------
# 11) SAVE CLEANED CSV
# -----------------------------
output_file = "water_quality_cleaned.csv"
df_clean.to_csv(output_file, index=False)

print(f"\nCleaned CSV saved as {output_file}")
files.download(output_file) 
