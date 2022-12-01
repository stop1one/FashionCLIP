import pandas as pd
import json

CAPTIONS_PATH = "./Dataset/captions/"

group = ['dress', 'shirt', 'toptee']

# mode: train, val, test
def preprocess_dataset(mode="train"):
    captions = []
    for i in range(3):
        with open(CAPTIONS_PATH+'cap.'+group[i]+'.'+mode+'.json', 'r') as c:
            captions += json.loads(c.read())

    # Reading example
    #print(captions[1])
    # {'target': 'B00BZ8GPVO', 'candidate': 'B008MTHLHQ', 'captions': ['is longer', 'is lighter and longer']}

    target = []
    candidate = []
    caption = []
    for i in range(len(captions)):
        for j in range(2):
            target.append(captions[i]["target"])
            candidate.append(captions[i]["candidate"])
            caption.append(captions[i]["captions"][j])
    
    return pd.DataFrame({
        "target": target,
        "candidate": candidate,
        "captions": caption
    })

# Test
if __name__ == '__main__':
    print(preprocess_dataset()["target"][0:5])
