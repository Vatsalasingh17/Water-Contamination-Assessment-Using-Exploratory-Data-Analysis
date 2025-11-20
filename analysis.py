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

# Set visual style for plots
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 4)

# -----------------------------
# 2) Upload CSV
# -----------------------------
print("Upload your water-quality CSV file:")
uploaded = files.upload()   # Allows user to manually upload CSV into Colab

filename = list(uploaded.keys())[0]
df = pd.read_csv(filename)  # Read CSV into DataFrame
print("\nLoaded:", filename)

# -----------------------------
# 3) Basic Structure
# -----------------------------
print("\n--- Shape ---")
print(df.shape)  # Display number of rows and columns

print("\n--- Columns & dtypes ---")
print(df.dtypes)  # Show data types of each column

# Detect numeric columns
numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

# Exclude target variable if numeric
if "is_safe" in numerical_cols:
    numerical_cols.remove("is_safe")

print("\nNumerical columns:", numerical_cols)
print("Categorical/Target:", "is_safe" if "is_safe" in df.columns else "None")

# Display first 5 rows for preview
display(df.head())

# -----------------------------
# 4) Missing Values
# -----------------------------
print("\n--- Missing Values ---")

# Create a table showing missing count and percentage
missing = pd.DataFrame({
    "missing_count": df.isnull().sum(),
    "missing_percent": (df.isnull().mean() * 100).round(2)
})
display(missing)

# Make a clean copy for processing
df_clean = df.copy()

# Impute numeric missing values using median
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

# Remove duplicate rows if present
if dups > 0:
    df_clean.drop_duplicates(inplace=True)
    print("Duplicates removed.")

# -----------------------------
# 6) Univariate Analysis
# -----------------------------
print("\n--- Univariate Analysis ---")

# Plot histogram and boxplot for each numeric column
for col in numerical_cols:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram + KDE
    sns.histplot(df_clean[col], kde=True, ax=ax[0])
    ax[0].set_title(f"Histogram — {col}")
    
    # Boxplot
    sns.boxplot(x=df_clean[col], ax=ax[1])
    ax[1].set_title(f"Boxplot — {col}")
    
    plt.show()

# Summary statistics table
display(df_clean[numerical_cols].describe().T)

# -----------------------------
# 7) Class Distribution (is_safe)
# -----------------------------
if "is_safe" in df_clean.columns:
    print("\n--- Class Distribution ---")

    # Frequency count of safe/unsafe
    display(df_clean["is_safe"].value_counts())

    # Countplot for class visualization
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df_clean["is_safe"])
    plt.title("is_safe Class Count")
    plt.show()

# -----------------------------
# 8) Bivariate Analysis (With is_safe)
# -----------------------------
if "is_safe" in df_clean.columns:
    print("\n--- Bivariate Analysis w.r.t is_safe ---")

    # Compare distribution of each numeric feature across safe/unsafe groups
    for col in numerical_cols:
        plt.figure(figsize=(10, 4))
        sns.boxplot(x="is_safe", y=col, data=df_clean)
        plt.title(f"{col} vs is_safe")
        plt.show()

    # Mean values for each class
    print("\nMean values grouped by class:")
    display(df_clean.groupby("is_safe")[numerical_cols].mean())

# -----------------------------
# 9) Correlation Analysis
# -----------------------------
print("\n--- Correlation Matrix ---")

# Compute correlation matrix
corr = df_clean.corr()

# Display heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Show strongest positive/negative correlations with target class
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

    # IQR-based outlier detection
    q1 = df_clean[col].quantile(0.25)
    q3 = df_clean[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    
    # Count values outside IQR range
    iqr_outliers[col] = (~df_clean[col].between(lower, upper)).sum()

    # Z-score based detection
    if df_clean[col].std() > 0:
        z = np.abs(stats.zscore(df_clean[col]))
        z_outliers[col] = (z > 3).sum()
    else:
        z_outliers[col] = 0

# Display outlier table
outlier_df = pd.DataFrame({"IQR_outliers": iqr_outliers, "Z_outliers": z_outliers})
display(outlier_df)

# -----------------------------
# 11) SAVE CLEANED CSV
# -----------------------------
output_file = "water_quality_cleaned.csv"

df_clean.to_csv(output_file, index=False)
print(f"\nCleaned CSV saved as {output_file}")

# Download cleaned file in Colab
files.download(output_file)
