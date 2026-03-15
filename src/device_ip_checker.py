import pandas as pd

class DeviceIPChecker:
    def __init__(self):
        # store known devices per user
        self.user_devices = {}

        # store known IPs per user
        self.user_ips = {}

        # known bad IPs (fraud blacklist)
        self.bad_ips = {
            "185.220.1.1",
            "185.220.1.2",
            "185.220.2.1",
            "185.220.3.1"
        }

    def train(self, data):
        """
        Learn each user's normal devices and IPs.
        data must have: user_id, device_id, ip_address columns
        """
        for user_id, group in data.groupby("user_id"):
            self.user_devices[user_id] = set(group["device_id"].unique())
            self.user_ips[user_id]     = set(group["ip_address"].unique())

        print(f"DeviceIPChecker trained on {len(self.user_devices)} users")

    def check(self, transaction):
        """
        Check if device or IP is suspicious.
        Returns: SUSPICIOUS or CLEAN
        """
        user_id   = int(transaction["user_id"].values[0])
        device_id = transaction["device_id"].values[0]
        ip        = transaction["ip_address"].values[0]

        # check 1 — is IP in blacklist?
        if ip in self.bad_ips:
            return "SUSPICIOUS"

        # check 2 — is device unknown for this user?
        if user_id in self.user_devices:
            if device_id not in self.user_devices[user_id]:
                return "SUSPICIOUS"

        # check 3 — is IP unknown for this user?
        if user_id in self.user_ips:
            if ip not in self.user_ips[user_id]:
                return "SUSPICIOUS"

        return "CLEAN"


if __name__ == "__main__":
    print("Testing DeviceIPChecker...")

    data = pd.read_csv("data/creditcard_augmented.csv", nrows=5000)

    checker = DeviceIPChecker()
    checker.train(data)

    sample = data.iloc[[0]]
    result = checker.check(sample)

    print(f"User ID   : {sample['user_id'].values[0]}")
    print(f"Device    : {sample['device_id'].values[0]}")
    print(f"IP        : {sample['ip_address'].values[0]}")
    print(f"Result    : {result}")

    # test with a fake suspicious transaction
    print("\n--- Testing suspicious transaction ---")
    sample2 = data.iloc[[0]].copy()
    sample2["device_id"] = "DEV_UNKNOWN_9999"
    sample2["ip_address"] = "185.220.1.1"
    result2 = checker.check(sample2)
    print(f"Device    : {sample2['device_id'].values[0]}")
    print(f"IP        : {sample2['ip_address'].values[0]}")
    print(f"Result    : {result2}")