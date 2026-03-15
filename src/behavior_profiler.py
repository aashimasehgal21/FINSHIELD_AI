import pandas as pd
import numpy as np

class BehaviorProfiler:
    def __init__(self):
        # store each user's stats separately
        self.user_stats = {}

        # fallback global stats (used when user is new / not seen before)
        self.global_avg = 0
        self.global_std = 1

    def train(self, data):
        """
        Learn each user's normal spending behavior.
        data must have: user_id, Amount columns
        """
        # save global stats as fallback
        self.global_avg = data["Amount"].mean()
        self.global_std = data["Amount"].std()

        # save per-user stats
        grouped = data.groupby("user_id")["Amount"].agg(["mean", "std", "count"])

        for user_id, row in grouped.iterrows():
            self.user_stats[user_id] = {
                "avg":   row["mean"],
                "std":   row["std"] if not pd.isna(row["std"]) else 0,
                "count": row["count"]
            }

        print(f"BehaviorProfiler trained on {len(self.user_stats)} users")

    def check_behavior(self, transaction):
        """
        Check if this transaction is unusual for THIS specific user.
        Returns: UNUSUAL or NORMAL
        """
        amount  = transaction["Amount"].values[0]
        user_id = int(transaction["user_id"].values[0]) if "user_id" in transaction.columns else None

        # get this user's stats — fallback to global if user not seen
        if user_id and user_id in self.user_stats:
            avg = self.user_stats[user_id]["avg"]
            std = self.user_stats[user_id]["std"]
        else:
            avg = self.global_avg
            std = self.global_std

        # if std is 0 (user has only 1 transaction) use global std
        if std == 0:
            std = self.global_std

        # flag if amount is more than 3 standard deviations above user's normal
        threshold = avg + 3 * std

        if amount > threshold:
            return "UNUSUAL"
        else:
            return "NORMAL"

    def get_user_profile(self, user_id):
        """
        Helper — see any user's profile.
        Useful for dashboard and explainability.
        """
        if user_id in self.user_stats:
            return self.user_stats[user_id]
        else:
            return {"avg": self.global_avg, "std": self.global_std, "count": 0}


if __name__ == "__main__":
    print("Testing BehaviorProfiler...")

    data = pd.read_csv("data/creditcard_augmented.csv")

    profiler = BehaviorProfiler()
    profiler.train(data)

    # test on first transaction
    sample = data.iloc[[0]]
    result = profiler.check_behavior(sample)
    print(f"Transaction Amount : {sample['Amount'].values[0]}")
    print(f"User ID            : {sample['user_id'].values[0]}")
    print(f"Behavior Status    : {result}")

    # show user profile
    uid = int(sample["user_id"].values[0])
    profile = profiler.get_user_profile(uid)
    print(f"User {uid} profile : avg={profile['avg']:.2f}, std={profile['std']:.2f}, txn_count={profile['count']}")