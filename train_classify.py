import torch
from torch import nn
from dataset import ModelNet40
from model import PVPNet
from torch.utils import data
from utils import shift_point_cloud, random_point_dropout, random_scale_point_cloud


gpu = torch.device("cuda:0")

net = PVPNet(n_cls=40, w=32, in_channel=6)
net.to(gpu)
dataset_path = "/home/zhang/dataset/modelnet40_ply_hdf5_2048"
param_save_path = "./params/pvpnet-w32-normal-n2048-aug-batch4-sgd.pth"
net.load_state_dict(torch.load(param_save_path))
batch_size = 4
accumulation = 8
# The real batchsize = batch_size x accumulation
lr = 0.001
min_lr = 0.00001
lr_update_step = 15
epoch = 300
optimizer = torch.optim.SGD(lr=lr, params=net.parameters(), momentum=0.9, weight_decay=1e-4)
# optimizer = torch.optim.Adam(lr=lr, params=net.parameters(), weight_decay=0)
loss_fn = nn.CrossEntropyLoss()
modelnet40_train, modelnet40_test = ModelNet40(dataset_path, "train"), ModelNet40(dataset_path, "test")
train_loader = data.DataLoader(modelnet40_train, shuffle=True, batch_size=batch_size)
test_loader = data.DataLoader(modelnet40_test, shuffle=False, batch_size=batch_size)


def update_lr(optimizer, gamma=0.5):
    lr = 0
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    lr = max(lr * gamma, min_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("lr update finished  cur lr: %.5f" % lr)


def processbar(current, totle):
    process_str = ""
    for i in range(int(20*current/totle)):
        process_str += "█"
    while len(process_str) < 20:
        process_str += " "
    return "%s|  %d / %d" % (process_str, current, totle)


if __name__ == '__main__':
    def evaluate():
        loss_val, process, correct = 0, 0, 0
        net.eval()
        with torch.no_grad():
            for pts, label in test_loader:
                out = net(pts.to(gpu))
                correct += (out.argmax(dim=1).cpu() == label).sum(dim=0).item()
                loss = loss_fn(out, label.to(gpu))
                loss_val += loss.item()
                process += pts.shape[0]
                print("\r测试进度：%s  本批loss:%.5f  当前精度: %.5f" % (processbar(process, len(modelnet40_test)), loss.item(), correct / process), end="")
        accuracy = correct / len(modelnet40_test)
        print("\n测试完毕  accuracy: %.5f" % accuracy)
        return accuracy


    def train():
        max_acc, iter_cnt = 0, 0
        for epoch_count in range(1, epoch + 1):
            net.train()
            loss_val, process, correct = 0, 0, 0
            optimizer.zero_grad()
            for pts, label in train_loader:
                iter_cnt += 1
                # print(pts.max(dim=1)[0], pts.min(dim=1)[0], pts[:, :, 3]**2+pts[:, :, 4]**2+pts[:, :, 5]**2)
                # 数据增强
                points = pts.data.numpy()
                points = random_point_dropout(points)
                points[:, :, 0:3] = random_scale_point_cloud(points[:, :, 0:3])
                points[:, :, 0:3] = shift_point_cloud(points[:, :, 0:3])
                pts = torch.Tensor(points)

                out = net(pts.to(gpu))
                correct += (out.argmax(dim=1).cpu() == label).sum(dim=0).item()
                loss = loss_fn(out, label.to(gpu)) / accumulation
                loss.backward()

                loss_val += loss.item()
                process += pts.shape[0]

                if iter_cnt % accumulation == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    print("\r进度：%s  本批loss:%.5f  当前精度: %.5f" % (processbar(process, len(modelnet40_train)), loss.item(), correct / process), end="")
            print("\nepoch:%d  loss:%.3f" % (epoch_count, loss_val))
            print("开始测试...")
            accuracy = evaluate()
            if max_acc < accuracy:
                max_acc = accuracy
                print("save...")
                torch.save(net.state_dict(), param_save_path)
                print("save finished !!!")
            if epoch_count % lr_update_step == 0:
                update_lr(optimizer, 0.5)

    # train()
    evaluate()