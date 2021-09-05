import numpy as np
import open3d as o3d
import torch
from dataset import ShapeNetCompletion
from model import PVPCompletionNet
from torch.utils import data
from torch import nn
from utils import square_distance


device = torch.device("cuda:0")
epoch = 251
point_num = 2048
lr = 0.001
min_lr = 0.00001
lr_update_step = 20
batch_size = 1
accumulation = 12
# l2 reg only for SGD
weight_decay = 1e-4

w = 32
n_add = 5

param_load_path = "./params/pvpnet-conpletion-w%d-normal-adam.pth" % w
param_save_path = "./params/pvpnet-conpletion-w%d-normal-adam.pth" % w
datset_path = "E:/shapenet_benchmark_v0_normal"
loss_fn = nn.CrossEntropyLoss()

net = PVPCompletionNet(n_add=n_add, w=w)
net.to(device)
net.load_state_dict(torch.load(param_load_path))
# optimizer = torch.optim.SGD(params=net.parameters(), lr=lr, weight_decay=weight_decay)
optimizer = torch.optim.Adam(params=net.parameters(), lr=lr, weight_decay=0)
cat = ['Airplane']
partseg_dataset_train = ShapeNetCompletion(datset_path, npoints=point_num, split='train', normal_channel=False, class_choice=cat)
partseg_dataset_test = ShapeNetCompletion(datset_path, npoints=point_num, split='test', normal_channel=False, class_choice=cat)
train_loader = data.DataLoader(dataset=partseg_dataset_train, batch_size=batch_size, shuffle=True)
# test dataset是不能随机取2048个点的，原本有几个点就用几个测，因此batch_size只能是1
test_loader = data.DataLoader(dataset=partseg_dataset_test, batch_size=1, shuffle=False)

print(len(partseg_dataset_train))


def processbar(current, totle):
    process_str = ""
    for i in range(int(20*current/totle)):
        process_str += "█"
    while len(process_str) < 20:
        process_str += " "
    return "%s|  %d / %d" % (process_str, current, totle)


def update_lr(optimizer, gamma=0.5):
    global lr
    lr = max(lr*gamma, min_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("lr update finished  cur lr: %.5f" % lr)


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.eye(num_classes)[y.cpu().data.numpy(), ].to(device)


if __name__ == '__main__':
    def look():
        grid_points = []
        d = 2 / w
        for u in range(w):
            for v in range(w):
                for k in range(w):
                    grid_points.append([-1 + d / 2 + u * d, -1 + d / 2 + k * d, 1 - d / 2 - v * d])
        grid_points = np.array(grid_points)
        net.eval()
        confidence_thresh = 0.3
        with torch.no_grad():
            for pts, label, cls_ in test_loader:
                pts, cls_, label = pts.to(device), cls_.to(device), label.to(device)
                pred = net(pts, to_categorical(cls_, 16))
                for i in range(pts.shape[0]):
                    confidence = pred[i][:, 0]
                    print(confidence)
                    pos_idx = (confidence > confidence_thresh)
                    neg_idx = (confidence <= confidence_thresh)
                    print(pos_idx.nonzero(as_tuple=False).view(-1).shape[0], neg_idx.nonzero(as_tuple=False).view(-1).shape[0])
                    grid_pc = o3d.PointCloud()
                    grid_pc.points = o3d.Vector3dVector(grid_points[pos_idx.cpu().numpy()])
                    croped_pc = o3d.PointCloud()
                    croped_pc.points = o3d.Vector3dVector(pts[i].cpu().numpy())

                    offset = torch.tanh(pred[i][:, 1:].view(-1, n_add, 3)) * (1/w) + torch.Tensor(grid_points).view(-1, 1, 3).to(device)
                    # offset = offset.view(-1, 3)
                    # 找出输入部分已经存在的格点
                    x_to_grid_dis = square_distance(pts[i].unsqueeze(0), torch.Tensor(grid_points).to(device).unsqueeze(0))
                    grid_idx = x_to_grid_dis.topk(k=1, dim=2, largest=False, sorted=False)[1].squeeze(2)[0].cpu()
                    # print(grid_idx)
                    # 把pos idx去掉已经存在的grid idx
                    pos_idx = torch.LongTensor(np.setdiff1d(pos_idx.nonzero(as_tuple=False).view(-1).cpu().numpy(), grid_idx.cpu().numpy()))
                    pos_points = offset[pos_idx].view(-1, 3).cpu().numpy()
                    pos_pc = o3d.PointCloud()
                    pos_pc.points = o3d.Vector3dVector(np.concatenate([pos_points, pts[i].cpu().numpy()], axis=0))

                    o3d.draw_geometries([croped_pc], window_name="crop pc", width=1000, height=800)
                    o3d.draw_geometries([grid_pc], window_name="grid pc", width=1000, height=800)
                    o3d.draw_geometries([pos_pc], window_name="pos pc", width=1000, height=800)


    def train():
        min_loss = 1e8
        for epoch_count in range(1, epoch + 1):
            net.train()
            loss_val, process, correct, iter_cnt = 0, 0, 0, 0
            optimizer.zero_grad()
            for pts, label, cls_ in train_loader:
                iter_cnt += 1
                # pts = pts.data.numpy()
                # pts[:, :, 0:3] = random_scale_point_cloud(pts[:, :, 0:3])
                # pts[:, :, 0:3] = shift_point_cloud(pts[:, :, 0:3])
                # pts = torch.Tensor(pts)
                pts, cls_, label = pts.to(device), cls_.to(device), label.to(device)
                confidence_loss, cd_loss = net.loss(pts, to_categorical(cls_, 16), label)
                # optimizer.zero_grad()
                cd_loss *= 100
                loss = confidence_loss + cd_loss
                loss.backward()
                # optimizer.step()
                loss_val += loss.item()
                process += pts.shape[0]
                if iter_cnt % accumulation == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    print("\r进度：%s  本批loss:%.5f  confidence loss: %.5f  cd loss: %.5f" % (processbar(process, len(partseg_dataset_train)), loss.item(), confidence_loss.item(), cd_loss.item()), end="")
                # print("\r进度：%s  本批loss:%.5f  当前精度: %.5f" % (processbar(process, len(partseg_dataset_train)), loss.item(), correct / (process*point_num)), end="")
            print("\nepoch:%d  loss:%.3f" % (epoch_count, loss_val))
            print("开始测试...")
            # accuracy, miou = evaluate()
            if min_loss > loss_val:
                min_loss = loss_val
                print("save...")
                torch.save(net.state_dict(), param_save_path)
                print("save finished !!!")
            if epoch_count % lr_update_step == 0:
                update_lr(optimizer, 0.5)

    # train()
    look()
    # evaluate()