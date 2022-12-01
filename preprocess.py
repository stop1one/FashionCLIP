import pandas as pd
import json
import config as Config

group = ['dress', 'shirt', 'toptee']

# mode: train, val, test
def preprocess_dataset(mode="train"):
    captions = []
    for i in range(3):
        with open(f"{Config.captions_path}/cap.{group[i]}.{mode}.json", 'r') as c:
            captions += json.loads(c.read())

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
    print(preprocess_dataset().sample(n=5))
