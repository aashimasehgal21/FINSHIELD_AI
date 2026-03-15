import pandas as pd
import networkx as nx

class GraphDetector:
    def __init__(self):
        self.graph = nx.Graph()
        self.fraud_nodes = set()

    def build_graph(self, data):
        """
        Build a graph where:
        - Nodes = users, devices, ip addresses
        - Edges = connections between them
        """
        print("Building fraud graph...")

        for _, row in data.iterrows():
            user_node   = f"user_{row['user_id']}"
            device_node = f"device_{row['device_id']}"
            ip_node     = f"ip_{row['ip_address']}"

            # add edges between user-device and user-ip
            self.graph.add_edge(user_node, device_node)
            self.graph.add_edge(user_node, ip_node)

            # if this transaction is fraud, mark all its nodes as suspicious
            if row["Class"] == 1:
                self.fraud_nodes.add(user_node)
                self.fraud_nodes.add(device_node)
                self.fraud_nodes.add(ip_node)

        print(f"Graph built — Nodes: {self.graph.number_of_nodes()}, Edges: {self.graph.number_of_edges()}")
        print(f"Fraud nodes identified: {len(self.fraud_nodes)}")

    def detect(self, transaction):
        """
        Check if this transaction's user/device/ip
        is connected to any known fraud node.
        Returns: FRAUD_NETWORK or CLEAN
        """
        user_node   = f"user_{int(transaction['user_id'].values[0])}"
        device_node = f"device_{transaction['device_id'].values[0]}"
        ip_node     = f"ip_{transaction['ip_address'].values[0]}"

        # direct fraud node check
        if user_node in self.fraud_nodes:
            return "FRAUD_NETWORK"
        if device_node in self.fraud_nodes:
            return "FRAUD_NETWORK"
        if ip_node in self.fraud_nodes:
            return "FRAUD_NETWORK"

        # neighbor check — is this user connected to any fraud node?
        if user_node in self.graph:
            neighbors = list(self.graph.neighbors(user_node))
            for neighbor in neighbors:
                if neighbor in self.fraud_nodes:
                    return "FRAUD_NETWORK"

        return "CLEAN"


if __name__ == "__main__":
    print("Testing GraphDetector...")

    data = pd.read_csv("data/creditcard_augmented.csv", nrows=5000)

    detector = GraphDetector()
    detector.build_graph(data)

    sample = data.iloc[[0]]
    result = detector.detect(sample)
    print(f"User ID     : {sample['user_id'].values[0]}")
    print(f"Device ID   : {sample['device_id'].values[0]}")
    print(f"IP Address  : {sample['ip_address'].values[0]}")
    print(f"Graph Result: {result}")