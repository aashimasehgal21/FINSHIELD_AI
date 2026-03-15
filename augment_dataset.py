import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

print("Loading creditcard.csv...")
df = pd.read_csv("data/creditcard.csv", low_memory=False)
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")

df["user_id"] = np.random.randint(1, 1001, len(df))

df["Time"] = pd.to_numeric(df["Time"], errors="coerce").fillna(0)
base_date = datetime(2024, 1, 1)
df["timestamp"] = df["Time"].apply(lambda s: base_date + timedelta(seconds=float(s)))

df["merchant_id"] = np.random.randint(1, 501, len(df))

def assign_device(row):
    if row["Class"] == 0:
        return f"DEV_{row['user_id']}" if random.random() > 0.05 else f"DEV_UNKNOWN_{random.randint(1000,9999)}"
    else:
        return f"DEV_{row['user_id']}" if random.random() > 0.40 else f"DEV_UNKNOWN_{random.randint(1000,9999)}"

df["device_id"] = df.apply(assign_device, axis=1)

def assign_ip(row):
    home_ip = f"192.168.{row['user_id'] % 255}.1"
    bad_ips = ["185.220.1.1", "185.220.1.2", "185.220.2.1", "185.220.3.1"]
    if row["Class"] == 0:
        return home_ip if random.random() > 0.05 else f"10.0.{random.randint(0,255)}.1"
    else:
        r = random.random()
        if r < 0.50:
            return home_ip
        elif r < 0.80:
            return random.choice(bad_ips)
        else:
            return f"10.0.{random.randint(0,255)}.1"

df["ip_address"] = df.apply(assign_ip, axis=1)

def assign_country(cls):
    if cls == 0:
        return "IN"
    else:
        return "IN" if random.random() < 0.50 else random.choice(["US", "NG", "RU", "CN", "RO"])

df["country"] = df["Class"].apply(assign_country)

print("Saving creditcard_augmented.csv...")
df.to_csv("data/creditcard_augmented.csv", index=False)
print(f"Done! Shape: {df.shape}")
print(df[["user_id", "timestamp", "merchant_id", "device_id", "ip_address", "country"]].head())