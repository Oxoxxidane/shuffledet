import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from dataset import MyDataset, trans_yolo
from model import Net
import os
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim

n_class = 12
trains_path = f'{os.getcwd()}\\train'
valida_path = f'{os.getcwd()}\\valida'
test_path = f'{os.getcwd()}\\test'


def bbox_iou(xa, ya, wa, ha, xb, yb, wb, hb):
    x1, y1, x2, y2 = xa - wa / 2, ya - ha / 2, xa + wa / 2, ya + ha / 2
    x3, y3, x4, y4 = xb - wb / 2, yb - hb / 2, xb + wb / 2, yb + hb / 2
    x5 = torch.maximum(x1, x3)
    y5 = torch.maximum(y1, y3)
    x6 = torch.minimum(x2, x4)
    y6 = torch.minimum(y2, y4)
    w = torch.maximum(x6 - x5, torch.Tensor([0]))
    h = torch.maximum(y6 - y5, torch.Tensor([0]))
    inter_area = w * h
    b1_area = (x2 - x1) * (y2 - y1)
    b2_area = (x4 - x3) * (y4 - y3)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou


def collate(batch):
    to_tensor = transforms.ToTensor()
    img_batch = []
    label_batch = []
    for img, label in batch:
        img_tensor = to_tensor(img)
        img_batch.append(img_tensor)
        label_batch.append(label)
    img_batch = torch.stack(img_batch, 0)
    return img_batch, label_batch


def batch_loss1(out_batch, label, img_size, class_num):
    loss = 0
    a, b, c = 0.1, 0.1, 1
    for out_map, lab in zip(torch.unbind(out_batch, dim=0), label):
        x_id, y_id = 0, 0
        for col_map in torch.unbind(out_map, dim=1):
            # print(col_map.size())
            for row_map in torch.unbind(col_map, dim=1):
                # print(row_map.size())
                x_step, y_step = 6 / img_size[1], 6 / img_size[0]
                x_cell, y_cell = x_step * (x_id + 0.5), y_step * (y_id + 0.5)
                class_list = torch.zeros(class_num)
                for anchor in lab:
                    class_id = int(anchor[0])
                    con = row_map[int(class_id) * 3]
                    h = row_map[int(class_id) * 3 + 1]
                    w = row_map[int(class_id) * 3 + 2]
                    x_lab = torch.tensor(float(anchor[1]))
                    y_lab = torch.tensor(float(anchor[2]))
                    h_lab = torch.tensor(float(anchor[4]))
                    w_lab = torch.tensor(float(anchor[3]))
                    loss = loss + a * F.sigmoid(con) * (((x_cell - x_lab) ** 2) + ((y_cell - y_lab) ** 2))
                    loss = loss + b * F.sigmoid(con) * bbox_iou(x_cell, y_cell, w, h, x_lab, y_lab, w_lab, h_lab)
                    if abs(x_cell - x_lab) <= x_step and abs(y_cell - y_lab) <= y_step:
                        class_list[class_id] = 1
                class_con = row_map.reshape(class_num, 3)
                class_con = class_con.permute(1, 0)
                loss = loss + c * F.cross_entropy(class_con[0], class_list)
                x_id = x_id + 1
            y_id = y_id + 1
    return loss


def batch_loss(out_batch, label, *, img_w, img_h, device, f=True):
    if f:
        a, b, c, d, e = 0, 5, 0, 10, 30
    else:
        a, b, c, d, e = 0, 0, 0, 0, 30
    # a, b, c, d, e = 0, 5, 0, 10, 1
    # 先生成x坐标图
    # 再生成y坐标图
    x_step, y_step = 6 / img_w, 6 / img_h
    x_cell, y_cell = torch.arange(x_step / 2, 1, x_step), torch.arange(y_step / 2, 1, y_step)
    x_cell = x_cell.repeat(img_h // 6, 1)
    y_cell = y_cell.repeat(img_h // 6, 1)
    y_cell = y_cell.permute(1, 0)
    x_cell = x_cell.to(device)
    y_cell = y_cell.to(device)
    sigmoid = nn.Sigmoid().to(device)
    bce = nn.BCEWithLogitsLoss().to(device)
    mse = nn.MSELoss().to(device)
    loss_sum = 0
    for out_map, lab in zip(torch.unbind(out_batch, dim=0), label):
        con = sigmoid(out_map[::3, :, :])
        h = out_map[1::3, :, :]
        w = out_map[2::3, :, :]
        pos_loss_all = 0
        box_loss_all = 0
        ce_p_loss_all = 0
        con_mask = torch.zeros(con.size()).to(device)
        x_id_a, y_id_a = [], []
        for anchor in lab:
            if not anchor:
                continue
            class_id = int(anchor[0])
            x_lab = torch.tensor(float(anchor[1]))
            y_lab = torch.tensor(float(anchor[2]))
            h_lab = torch.tensor(float(anchor[4]))
            w_lab = torch.tensor(float(anchor[3]))
            x_loss = (x_cell - x_lab) ** 2
            y_loss = (y_cell - y_lab) ** 2
            pos_loss_anchor = torch.sum(torch.mul(x_loss + y_loss, con[class_id]))
            pos_loss_all = pos_loss_all + pos_loss_anchor

            x_id = int(torch.floor(x_lab / x_step))
            y_id = int(torch.floor(y_lab / y_step))
            h_dec = sigmoid(h[class_id, y_id, x_id])
            w_dec = sigmoid(w[class_id, y_id, x_id])
            # 求预测框与目标框最小高宽
            h_min = torch.min(h_dec, h_lab)
            w_min = torch.min(w_dec, w_lab)
            # 计算交集面积
            s_min = h_min * w_min
            # 计算并集面积
            area = h_dec * w_dec + h_lab * w_lab - s_min
            iou_anchor = 1 - s_min / area
            box_loss_all = box_loss_all + iou_anchor

            con_mask[class_id, y_id, x_id] = 1
            ce_p_loss_all = bce(con[class_id, y_id, x_id], torch.tensor(1.).to(device))
        ce_loss = bce(con_mask, con)
        loss_map = a * pos_loss_all + b * box_loss_all + c * ce_loss + \
                   d * ce_p_loss_all + e * mse(con, torch.tensor(0.).to(device))
        loss_sum = loss_sum + loss_map
    return loss_sum


def main():
    torch.manual_seed(114514)
    print("CUDA state is:")
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 32
    classes = 12
    train_set = MyDataset(path=trains_path, transform=trans_yolo)
    trains_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    net = Net(classes * 3).to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    print(net)
    for epoch in range(2):
        for img, label in trains_loader:
            # print(img.size())
            # print(label)
            forward = net(img.to(device))
            loss = batch_loss(forward, label, img_w=360, img_h=360, device=device, f=False)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(epoch, loss)

    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    for epoch in range(30):
        for img, label in trains_loader:
            # print(img.size())
            # print(label)
            forward = net(img.to(device))
            loss = batch_loss(forward, label, img_w=360, img_h=360, device=device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(epoch, loss)
    torch.save(net.state_dict(), 'model0_l2.mdl')
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    for epoch in range(70):
        for img, label in trains_loader:
            # print(img.size())
            # print(label)
            forward = net(img.to(device))
            loss = batch_loss(forward, label, img_w=360, img_h=360, device=device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(epoch, loss)
    torch.save(net.state_dict(), 'model1_l2.mdl')


if __name__ == '__main__':
    main()
