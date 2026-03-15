import pandas as pd
import requests
import time

DELAY      = 0.3
SERVER_URL = "http://localhost:8000/predict"

def clean_decision(val):
    val = str(val).upper()
    if "BLOCK" in val:
        return "BLOCK"
    elif "ALLOW" in val:
        return "ALLOW"
    else:
        return "REVIEW"

def simulate():
    print("Loading augmented dataset...")
    df = pd.read_csv("data/creditcard_augmented.csv")

    # 40 normal + 20 fraud rows
    normal_rows = df[df["Class"] == 0].sample(40, random_state=42)
    fraud_count = min(20, len(df[df["Class"] == 1]))
    fraud_rows  = df[df["Class"] == 1].sample(fraud_count, random_state=42)

    mixed = pd.concat([normal_rows, fraud_rows]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Sending {len(mixed)} transactions (40 normal + {fraud_count} fraud)...")
    print("=" * 70)

    for i, row in mixed.iterrows():
        transaction = row.to_dict()
        transaction["timestamp"] = str(transaction["timestamp"])

        try:
            response = requests.post(SERVER_URL, json=transaction)

            if response.status_code == 200:
                result   = response.json()
                risk     = result.get("risk_level", "?")
                decision = clean_decision(result.get("decision", "REVIEW"))
                prob     = result.get("fraud_probability", 0)

                icon = "🔴" if risk == "HIGH" else "🟡" if risk == "MEDIUM" else "🟢"

                print(f"{icon} Txn {i+1:03d} | Amount: {row['Amount']:>8.2f} | "
                      f"User: {row['user_id']:>4} | "
                      f"Fraud prob: {prob:.3f} | "
                      f"Risk: {risk:<6} | Decision: {decision}")
            else:
                print(f"❌ Txn {i+1:03d} | Server error: {response.status_code}")

        except requests.exceptions.ConnectionError:
            print("❌ Server not running! Start: uvicorn app.main:app --reload")
            break

        time.sleep(DELAY)

    print("=" * 70)
    print("Simulation complete!")

if __name__ == "__main__":
    simulate()