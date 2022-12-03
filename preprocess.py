import pandas as pd
import json
import config as Config
import numpy as np

# mode: train, val, test
def preprocess_dataset(mode="train"):
    if mode == "test":
        splits = []
        for group in ["test", "train", "val"]:
            with open(f"{Config.split_path}/split.shirt.{group}.json", 'r') as s:
                splits += json.loads(s.read())

        caption = [""]*len(splits)
        df = pd.DataFrame({
            "target": splits,
            "candidate": splits,
            "caption": caption
        })
        
    else:
        captions = []
        with open(f"{Config.captions_path}/cap.shirt.{mode}.json", 'r') as c:
            captions = json.loads(c.read())

        target = []
        candidate = []
        caption = []
        for i in range(len(captions)):
            for j in range(2):
                target.append(captions[i]["target"])
                candidate.append(captions[i]["candidate"])
                caption.append(captions[i]["captions"][j])
        
        df = pd.DataFrame({
            "target": target,
            "candidate": candidate,
            "caption": caption
        })

    # Randomly shuffle
    return df.sample(frac=1).reset_index(drop=True)

# Test
if __name__ == '__main__':
    dataset = preprocess_dataset()
    print(dataset.sample(n=5))
    print(dataset['target'].values)
    print(dataset['candidate'].values)
    print(np.concatenate((dataset['target'].values, dataset['candidate'].values)))
    test = preprocess_dataset("test")
    print(test.sample(n=5))
