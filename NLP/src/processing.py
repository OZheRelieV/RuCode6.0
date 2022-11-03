import os
import PIL
import pandas as pd
import matplotlib.pylab as plt
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def make_submit(filename, preds):
    if(os.path.exists(f"{filename}.csv")):
        print(f"File '{filename}.csv' already exist. Are you want to rewrite it? [yes/no]")
        ans = input(">")
        if((ans != "yes") and (ans != "no")):
            raise ValueError("Only 'yes/no' are accessible")
        else:
            if(ans == "yes"):
                pd.Series(preds, name=None).to_csv(f"{filename}.csv", index=False, header=False)
                print("=====SUBMITION DONE=====")
            else:
                print("Define new name")
                new_name = input(">")
                make_submit(new_name, preds)
    else:
        pd.Series(preds, name=None).to_csv(f"{filename}.csv", index=False, header=False)
        print("=====SUBMITION DONE=====")


def check_images_amount():
    images_amount = {}
    for path in os.listdir():
        if(os.path.isdir(path) and (("train" in path) or ("test" in path))):
            if(path == "train"):
                count = 0
                for train_path in os.listdir(path):
                    count += len(os.listdir(path +"//" + train_path))
                images_amount[path] = count
            else:
                images_amount[path.split('_')[1]] = len(os.listdir(path))
    return images_amount


def visualize_distribution(data, mapping):
    data_to_print = {mapping[d[1]] : d[0]
                     for d in zip(data["target"].value_counts().values, data["target"].value_counts().index)}
    plt.bar(list(data_to_print.keys()), list(data_to_print.values()), width=0.4)


class CarDataset(Dataset):
    def __init__(self, annotation_file, img_dir, new_size, mode="train"):
        super().__init__()
        self.img_labels = annotation_file
        self.img_dir = img_dir
        self.mode = mode
        self.new_size = new_size
        
    def __len__(self):
        return len(self.img_labels)
    
    def _load_sample(self, file):
        image = PIL.Image.open(file)
        image.load()
        return image
    
    def __getitem__(self, idx):
        transform = transforms.Compose([
            transforms.Resize(size=(self.new_size, self.new_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
            ])
        image = self._load_sample(os.path.join(self.img_dir + '//' + self.img_labels.iloc[idx, 0]))
        image = transform(image)
        label = self.img_labels.iloc[idx, 1]
        if self.mode != "test":
            return image, label
        else:
            return image


def train_val_split(aanotation_file):
    x_train, x_valid, y_train, y_valid = train_test_split(aanotation_file["img_path"], aanotation_file["target"], test_size=0.25,
                                                          shuffle=True, stratify=aanotation_file["target"])
    train = pd.DataFrame(columns=["img_path", "target"], data=[[tr[0], tr[1]] for tr in zip(x_train, y_train)])
    valid = pd.DataFrame(columns=["img_path", "target"], data=[[val[0], val[1]] for val in zip(x_valid, y_valid)])

    return train, valid