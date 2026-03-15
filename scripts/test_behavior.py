import pandas as pd
import sys
import os

# project root path add
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.behavior_profiler import BehaviorProfiler

print("Step 1: Loading dataset")

data = pd.read_csv("data/creditcard.csv", nrows=2000)

print("Step 2: Training behavior profiler")

profiler = BehaviorProfiler()
profiler.train(data)

print("Step 3: Testing transaction behavior")

sample = data.iloc[[0]]

result = profiler.check_behavior(sample)

print("Behavior Status:", result)