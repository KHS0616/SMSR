"""
SMSR Dataset 코드

Writer : KHS0616
Last Update : 2021-11-08
"""
import torch
from torchvision.transforms import Compose, RandomCrop, ToTensor, Resize

import os
from PIL import Image

def check_image(file_name):
    ext = (".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG")
    return True if file_name.endswith(ext) else False

class TrainDatasets(torch.utils.data.Dataset):
    def __init__(self):
        # 파일 경로 설정 및 리스트 작성
        self.file_path_HR = "../Image/div2k_hr"
        self.file_list_HR = [x for x in os.listdir(self.file_path_HR) if check_image(x)]

        self.file_path_LR = "../Image/div2k_lrx2"
        self.file_list_LR = [x for x in os.listdir(self.file_path_LR) if check_image(x)]        

        # 옵션 설정
        self.n_colors = 3
        self.rgb_range = 255
        self.train = True
        self.no_augment = False
        self.patch_size = 96

        self.transform_HR = Compose([
            ToTensor()
        ])

        self.transform_CROP = Compose([
            RandomCrop(192)
        ])

        self.transform_LR = Compose([
            Resize((96,96)),
            ToTensor()
        ])

    def __getitem__(self, idx):
        HR = Image.open(os.path.join(self.file_path_HR, self.file_list_HR[idx]))
        croped_tensor = self.transform_CROP(HR)
        HR_tensor = self.transform_HR(croped_tensor) * 255
        LR_tensor = self.transform_LR(croped_tensor) * 255

        return LR_tensor, HR_tensor, self.file_list_HR[idx]

    def __len__(self):
        return len(self.file_list_HR)