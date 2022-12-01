import os
import cv2
import torch
import albumentations as Aug

from config import Config


class FashionIQDataset(torch.utils.data.Dataset):
    """
    image_filenames and captions must have the same length;
    so, if there are multiple captions for each image, the image_filenames must have repetitive file names
    self: image_filenames, captions, encoded_captions, transforms
    """
    def __init__(self, image_filenames, captions, tokenizer, transforms):
        self.image_filenames = image_filenames # tuple (imgfname1, imgfname2)
        self.captions = list(captions)
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=Config.max_length
        )   # return: dictionary
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        images = []
        img_vec = []     # 0: target, 1: candidate
        for i in range(2):
            images.append(cv2.imread(f"{Config.image_path}/{self.image_filenames[idx][0]}"))
            images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            images[i] = self.transforms(image=images[i])['image']
            img_vec.append(torch.tensor(images[i]).permute(2, 0, 1).float())
        item['image'] = img_vec[0] - img_vec[1]     # target - candidate
        item['caption'] = self.captions[idx]

        return item

    def __len__(self):
        return len(self.captions)


def get_transforms(mode="train"):
    if mode == "train":
        return Aug.Compose(
            [
                # Resize function makes image have the same size
                Aug.Resize(Config.size, Config.size, always_apply=True),
                Aug.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return Aug.Compose(
            [
                Aug.Resize(Config.size, Config.size, always_apply=True),
                Aug.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )

    