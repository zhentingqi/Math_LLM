import json
import os


src = "../data/multiarith/test.json"
tgt = "../data/multiarith/test_with_ids.json"

with open(src, "r") as f_in, open(tgt, "w") as f_out:
    src_data = json.load(f_in)
    tgt_data = []
    for i, item in enumerate(src_data):
        item["id"] = i
        tgt_data.append(item)
    json.dump(tgt_data, f_out)