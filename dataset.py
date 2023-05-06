from torch.utils.data import Dataset
import os
import cv2

trains_path = f'{os.getcwd()}\\trains'
valida_path = f'{os.getcwd()}\\valida'
test_path = f'{os.getcwd()}\\test'


def trans_yolo(img_path, label_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (360, 360), interpolation=cv2.INTER_LINEAR)
    with open(label_path, "r") as file:
        label = []
        for line in file:
            label.append(line.split())
    return img, label


class MyDataset(Dataset):
    def __init__(self, path, transform):
        self.img_path = path + '\\img\\'
        self.label_path = path + '\\label\\'
        self.img_list = os.listdir(self.img_path)
        self.label_list = [os.path.splitext(file_name)[0] + '.txt' for file_name in self.img_list]
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.transform(self.img_path + self.img_list[index],
                                    self.label_path + self.label_list[index])
        return img, label

    def __len__(self):
        return len(self.img_list)


def main():
    test_set = MyDataset(path=test_path, transform=trans_yolo)
    print(test_set.img_path)
    print(test_set.img_list)
    print(test_set.label_list)
    img, label = test_set.__getitem__(3)
    print(img)
    print(label)


if __name__ == '__main__':
    main()
