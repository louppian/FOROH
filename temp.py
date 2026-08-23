import json
from pathlib import Path

p = Path(r"D:\FOROH\Dataset\LIMUC\cross_validation_folds_train_val_info\fold_0\train.json")
with open(p) as f:
    data = json.load(f)
print(f"type: {type(data).__name__}")
if isinstance(data, list):
    print(f"len: {len(data)}")
    print(f"first 3: {data[:3]}")
elif isinstance(data, dict):
    keys = list(data.keys())
    print(f"len: {len(keys)}")
    print(f"first 3 keys: {keys[:3]}")
    print(f"first 3 values: {[data[k] for k in keys[:3]]}")