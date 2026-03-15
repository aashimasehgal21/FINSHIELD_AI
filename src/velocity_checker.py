import pandas as pd
from datetime import timedelta

class VelocityChecker:
    def __init__(self):
        # thresholds — how many transactions in what time = suspicious
        self.limits = {
            "1min":  3,   # more than 3 in 1 minute = suspicious
            "10min": 10,  # more than 10 in 10 minutes = suspicious
            "1hour": 20   # more than 20 in 1 hour = suspicious
        }
        # store transaction history per user
        self.user_history = {}

    def train(self, data):
        """
        Load all past transactions into memory per user.
        data must have: user_id, timestamp columns
        """
        data["timestamp"] = pd.to_datetime(data["timestamp"])

        for user_id, group in data.groupby("user_id"):
            self.user_history[user_id] = list(group["timestamp"].sort_values())

        print(f"VelocityChecker loaded history for {len(self.user_history)} users")

    def check(self, transaction):
        """
        Check if this user is making too many transactions recently.
        Returns: SUSPICIOUS or NORMAL
        """
        user_id   = int(transaction["user_id"].values[0])
        txn_time  = pd.to_datetime(transaction["timestamp"].values[0])

        # if user not seen before — no history — return normal
        if user_id not in self.user_history:
            return "NORMAL"

        history = self.user_history[user_id]

        # count transactions in each time window
        count_1min  = sum(1 for t in history if txn_time - timedelta(minutes=1)  <= t <= txn_time)
        count_10min = sum(1 for t in history if txn_time - timedelta(minutes=10) <= t <= txn_time)
        count_1hour = sum(1 for t in history if txn_time - timedelta(hours=1)    <= t <= txn_time)

        # check against limits
        if count_1min > self.limits["1min"]:
            return "SUSPICIOUS"
        if count_10min > self.limits["10min"]:
            return "SUSPICIOUS"
        if count_1hour > self.limits["1hour"]:
            return "SUSPICIOUS"

        return "NORMAL"


if __name__ == "__main__":
    print("Testing VelocityChecker...")

    data = pd.read_csv("data/creditcard_augmented.csv", nrows=5000)

    checker = VelocityChecker()
    checker.train(data)

    sample = data.iloc[[0]]
    result = checker.check(sample)

    print(f"User ID    : {sample['user_id'].values[0]}")
    print(f"Timestamp  : {sample['timestamp'].values[0]}")
    print(f"Result     : {result}")