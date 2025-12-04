import boto3
import pandas as pd
import numpy as np
import joblib
import os
import zipfile
import py7zr
import gc
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor

# --- Configuration ---
BUCKET_NAME = 'favorita-project-data'
RANDOM_STATE = 42
S3_PREFIX = 'models/'

s3 = boto3.client('s3')

print("1. Finding and Extracting Data...")
# Files inside 'models/' folder
files_in_s3 = [
    'models/train.csv.7z.zip', 'models/items.csv.7z.zip',
    'models/stores.csv.7z.zip', 'models/oil.csv.7z.zip',
    'models/holidays_events.csv.7z.zip', 'models/transactions.csv.7z.zip'
]

def extract_archive(filename):
    if filename.endswith('.zip'):
        try:
            with zipfile.ZipFile(filename, 'r') as z:
                z.extractall()
                for f in z.namelist():
                    if f.endswith('.7z') and not f.startswith('__'):
                        with py7zr.SevenZipFile(f, mode='r') as seven_z:
                            seven_z.extractall()
                        os.remove(f)
            os.remove(filename)
        except Exception: pass

for s3_key in files_in_s3:
    local_name = os.path.basename(s3_key)
    target_csv = local_name.replace('.7z.zip', '')
    if not os.path.exists(target_csv):
        print(f"   Fetching {local_name}...")
        try:
            s3.download_file(BUCKET_NAME, s3_key, local_name)
            extract_archive(local_name)
        except: pass

if os.path.exists("oil.scv"): os.rename("oil.scv", "oil.csv")

print("\n2. Loading Auxiliary Data...")
items = pd.read_csv("items.csv")
stores = pd.read_csv("stores.csv")
oil = pd.read_csv("oil.csv")
holidays = pd.read_csv("holidays_events.csv")
transactions = pd.read_csv("transactions.csv")

# Date conversion
oil["date"] = pd.to_datetime(oil["date"])
holidays["date"] = pd.to_datetime(holidays["date"])
transactions["date"] = pd.to_datetime(transactions["date"])

print("3. Sampling Training Data (Aggressive)...")
# ULTRA-AGGRESSIVE SAMPLING for 1GB RAM
rng = np.random.default_rng(RANDOM_STATE)
sample_stores = rng.choice(stores["store_nbr"].unique(), size=3, replace=False)
sample_items = rng.choice(items["item_nbr"].unique(), size=50, replace=False)

print(f"   Filtering for {len(sample_stores)} stores and {len(sample_items)} items...")

chunks = []
for chunk in pd.read_csv("train.csv", chunksize=500000):
    # Filter
    filtered = chunk[
        (chunk['store_nbr'].isin(sample_stores)) & 
        (chunk['item_nbr'].isin(sample_items))
    ].copy()
    
    if not filtered.empty:
        # Optimization immediately after read
        filtered['date'] = pd.to_datetime(filtered['date'])
        filtered['unit_sales'] = filtered['unit_sales'].astype('float32')
        filtered['onpromotion'] = filtered['onpromotion'].fillna(False).astype(bool)
        chunks.append(filtered)

train = pd.concat(chunks, ignore_index=True)
del chunks
gc.collect()

print(f"   Final dataset shape: {train.shape}")
if train.empty:
    print("Error: Sample is empty. Increase sample size slightly.")
    exit()

print("4. Merging Data...")
train = train.merge(items, on="item_nbr", how="left")
train = train.merge(stores, on="store_nbr", how="left")
train = train.merge(transactions, on=["date", "store_nbr"], how="left")
train = train.merge(oil, on="date", how="left")

# Simple holiday processing to save memory
holidays = holidays[['date', 'type']].rename(columns={'type': 'holiday_type'})
train = train.merge(holidays, on="date", how="left")

# Fill N/As
train['dcoilwtico'] = train['dcoilwtico'].ffill().bfill()
train['transactions'] = train['transactions'].fillna(0)

# Target
train['log_unit_sales'] = np.log1p(train['unit_sales'].clip(lower=0))

# Features
print("5. Training...")
feature_cols = ['store_nbr', 'item_nbr', 'dcoilwtico', 'transactions', 'onpromotion']
X = train[feature_cols].fillna(0)
y = train['log_unit_sales']

# Simple Model
model = LGBMRegressor(n_estimators=50, n_jobs=1) 
model.fit(X, y)
print("   Training complete.")

print("6. Uploading...")
joblib.dump(model, 'sales_forecast_pipeline.pkl')
s3.upload_file('sales_forecast_pipeline.pkl', BUCKET_NAME, 'models/sales_forecast_pipeline.pkl')
print("SUCCESS!")