import cv2
from model import Net
from dataset import MyDataset, trans_yolo
import os
import torch.utils.data as data
import torch
from train import collate
import torch.nn as nn
import numpy as np

trains_path = f'{os.getcwd()}\\train'
valida_path = f'{os.getcwd()}\\valid'
test_path = f'{os.getcwd()}\\test'


def draw_box(img, lab, *, w_img, h_img):
    img = np.ascontiguousarray(img)
    for anchor in lab:
        class_id = anchor[0]
        x_lab = float(anchor[1])
        y_lab = float(anchor[2])
        h_lab = float(anchor[4])
        w_lab = float(anchor[3])
        x, y, w, h = x_lab * w_img, y_lab * h_img, w_lab * w_img, h_lab * h_img
        x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), [255, 255, 255], 1)
        img = cv2.putText(img, str(class_id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                          0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def draw_box_det(img, out_map, *, img_w, img_h, th, device):
    x_step, y_step = 6 / img_w, 6 / img_h
    x_cell, y_cell = torch.arange(x_step / 2, 1, x_step).to(device), torch.arange(y_step / 2, 1, y_step).to(device)
    sigmoid = nn.Sigmoid().to(device)

    con = out_map[::3, :, :]
    h = sigmoid(out_map[1::3, :, :])
    w = sigmoid(out_map[2::3, :, :])
    # print(con)
    max_value = torch.max(con)
    th = max_value * th
    for i, (class_, c, h, w) in enumerate(zip(torch.unbind(out_map, dim=0), torch.unbind(con, dim=0),
                                              torch.unbind(h, dim=0), torch.unbind(w, dim=0))):
        center = torch.unbind(torch.nonzero(c > th), dim=0)
        if not center:
            continue
        for pos in center:
            conf = c[pos.cpu().numpy()[0], pos.cpu().numpy()[1]].cpu().detach().numpy()
            x = x_cell[pos.cpu().numpy()[1]].cpu().numpy()
            y = y_cell[pos.cpu().numpy()[0]].cpu().numpy()
            h1 = h[pos.cpu().numpy()[0], pos.cpu().numpy()[1]].cpu().detach().numpy()
            w1 = w[pos.cpu().numpy()[0], pos.cpu().numpy()[1]].cpu().detach().numpy()
            print(i, conf, x, y, h1, w1, pos.cpu().numpy())

            img = draw_box(img, [[i, x, y, w1, h1]], w_img=img_w, h_img=img_h)
    return img


def main():
    torch.manual_seed(114514)
    print("CUDA state is:")
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 16
    classes = 12
    w_img, h_img = 360, 360
    train_set = MyDataset(path=trains_path, transform=trans_yolo)
    trains_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate)
    net = Net(classes * 3).to(device)
    net.load_state_dict(torch.load('model1_l2.mdl'))
    net.eval()
    for img_batch, label in trains_loader:
        forward = net(img_batch.to(device))
        for img, lab, out in zip(torch.unbind(img_batch, dim=0), label, torch.unbind(forward, dim=0)):
            # print(img.size())
            image = img.permute(1, 2, 0).numpy()
            image0 = draw_box(image, lab, w_img=w_img, h_img=h_img)
            cv2.imshow("Test Image", image0)
            image1 = draw_box_det(image, out, img_w=w_img, img_h=h_img, th=0.5, device=device)
            cv2.imshow("Det Image", image1)
            # print(out.size())
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
