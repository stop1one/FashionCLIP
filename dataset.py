import os
import cv2
import torch
import albumentations as Aug
import config as Config


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

        target_images = cv2.imread(f"{Config.image_path}/{self.image_filenames[idx][0]}.png")
        target_images = cv2.cvtColor(target_images, cv2.COLOR_BGR2RGB)
        target_images = self.transforms(image=target_images)['image']
        item['target_image'] = torch.tensor(target_images).permute(2, 0, 1).float()

        candidate_images = cv2.imread(f"{Config.image_path}/{self.image_filenames[idx][1]}.png")
        candidate_images = cv2.cvtColor(candidate_images, cv2.COLOR_BGR2RGB)
        candidate_images = self.transforms(image=candidate_images)['image']
        item['candidate_image'] = torch.tensor(candidate_images).permute(2, 0, 1).float()
        
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

# Dataset Load Test
if __name__ == "__main__":
    from preprocess import preprocess_dataset
    df = preprocess_dataset()
    print("complete to load dataset")
    image_filenames = list(zip(df["target"].values, df["candidate"].values))
    print(f"image: {image_filenames[0][0]}")
    idx = 0
    test_img = cv2.imread(f"{Config.image_path}/{image_filenames[idx][0]}.png")
    #cv2.imshow("test-image", test_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    transforms = get_transforms("train")
    target_images = cv2.imread(f"{Config.image_path}/{image_filenames[idx][0]}.png")
    target_images = cv2.cvtColor(target_images, cv2.COLOR_BGR2RGB)
    target_images = transforms(image=target_images)['image']
    target = torch.tensor(target_images).permute(2, 0, 1).float()
    candidate_images = cv2.imread(f"{Config.image_path}/{image_filenames[idx][1]}.png")
    candidate_images = cv2.cvtColor(candidate_images, cv2.COLOR_BGR2RGB)
    candidate_images = transforms(image=candidate_images)['image']
    candidate = torch.tensor(candidate_images).permute(2, 0, 1).float()
    print(target.size())
    print(candidate.size())
    tc = torch.stack((target, candidate), dim=0)
    print(tc.size())
    print(tc[0].size())
    print(tc[1].size())    
