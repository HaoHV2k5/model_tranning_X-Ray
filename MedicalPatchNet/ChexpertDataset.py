from torch.utils.data import Dataset
from utilFunc import readCSV
import argParser as ARG
from torchvision.transforms import v2
from PIL import Image
import torchvision.transforms.functional
import torch
import os

TEST = "test"
TRAIN = "train"
VALIDATE = "validate"

POSSIBLE_SPLITS = [
    TEST,
    TRAIN,
    VALIDATE,
]

# Thư mục chứa train.csv / val.csv / test.csv / images/ (đặt bằng biến môi trường CHEXPERT_ROOT)
def _find_base_path():
    env_root = os.environ.get("CHEXPERT_ROOT")
    if env_root and os.path.exists(env_root):
        return env_root if env_root.endswith("/") else env_root + "/"
    for candidate in ["../chexpert_subset", "./chexpert_subset", "chexpert_subset"]:
        if os.path.exists(candidate):
            return candidate if candidate.endswith("/") else candidate + "/"
    return "./chexpert_subset/"

BASE_PATH = _find_base_path()

META_DICT = {
    TRAIN:"train.csv",
    VALIDATE:"val.csv",
    TEST:"test.csv",
}

META_LABEL_CUT = 5
META_LABEL_CUT_DICT = {
    TRAIN:5,
    VALIDATE:5,
    TEST:5,
}

def shiftVisualChexpert(csvCont): # align visualCheXpert label with the normal chexpert training label order
    labelCut = 5
    for line in csvCont:
        lastEl = line.pop()
        line.insert(labelCut,lastEl)
    print("SHIFTED VISUAL CHEXPERT TRAINING FILE")


class ClassificationChexpertDataset(Dataset):
    def __init__(self, usedSplit, augementImg=False, imgSize=None, cache_in_ram=True):
        self.usedSplit = usedSplit
        self.augementImg = augementImg
        self.imgSize = imgSize if imgSize is not None else (ARG.IMG_SIZE if ARG.IMG_SIZE is not None else 512)
        self.cache_in_ram = cache_in_ram
        self.ram_cache = {}
        metaFile = BASE_PATH + META_DICT[self.usedSplit]
        #head,*data = readCSV(metaFile)
        csvCont = readCSV(metaFile)
        if META_DICT[self.usedSplit] == "train_visualCheXbert.csv": shiftVisualChexpert(csvCont) #shift dataset
        head,*data = csvCont

        labelCut = META_LABEL_CUT_DICT[usedSplit]
        self.labelNames = head[labelCut:]

        getTotalPath = lambda relativePath: BASE_PATH + relativePath.replace("CheXpert-v1.0/","")

        labelDict = { #set unknown to 0 => there is no unknow in in the visual chexpert label
            "": 0,
            "1.0": 1,
            "0.0": 0,
            "-1.0": 0,
        }
        #labelToInt = lambda lst: [int(float(x)) for x in lst]
        labelToInt = lambda lst: [labelDict[x] for x in lst]
        self.itemList = [(getTotalPath(x[0]),labelToInt(x[labelCut:])) for x in data]
        self.augmentTransforms = v2.Compose([
            v2.RandomResizedCrop(size=(self.imgSize,self.imgSize),scale=(0.5,1),antialias=True),
            v2.RandomRotation(degrees=5),
            v2.ColorJitter(brightness=0.3),
        ])
        self.normalTransform = v2.Compose([
            v2.Resize(size=(self.imgSize,self.imgSize),antialias=True)
        ])

    def applyAugment(self,img):
        if not self.augementImg:
            return self.normalTransform(img)
        img = self.augmentTransforms(img)
        return img
    
    def getLabelNames(self):
        return self.labelNames
    
    def __len__(self):
        return len(self.itemList)
    
    def __getitem__(self, idx):
        imgPath,label = self.itemList[idx]
        img = self.getImg(imgPath)
        img = self.applyAugment(img)

        label = torch.tensor(label,dtype=torch.float32)

        return img,label

    def getImg(self,imgPath):
        if self.cache_in_ram and imgPath in self.ram_cache:
            return self.ram_cache[imgPath]

        try:
            import torchvision.io as tv_io
            img = tv_io.read_image(imgPath, tv_io.ImageReadMode.GRAY).float() / 255.0
        except Exception:
            img = Image.open(imgPath).convert("L")
            img = torchvision.transforms.functional.to_tensor(img)

        if self.cache_in_ram:
            self.ram_cache[imgPath] = img
        return img