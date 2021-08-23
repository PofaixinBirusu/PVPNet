import numpy as np
import open3d as o3d
import torch
from torch import nn
from utils import square_distance
from dataset import ModelNet40
from torch.utils import data


gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PVPNet(nn.Module):
    def __init__(self, in_channel=3, w=32, n_cls=40):
        super(PVPNet, self).__init__()
        self.grid_points = []
        self.w, self.n_cls = w, n_cls
        d = 2 / w
        for u in range(w):
            for v in range(w):
                for k in range(w):
                    self.grid_points.append([-1+d/2+u*d, -1+d/2+k*d, 1-d/2-v*d])
        self.grid_points = torch.Tensor(self.grid_points)

        self.point2vec = nn.Sequential(
            nn.Conv1d(in_channel, 64, kernel_size=1, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=1, stride=1),
            # nn.BatchNorm1d(64),
            # nn.ReLU(inplace=True)
        )

        # 3DCNN
        self.conv1 = nn.Sequential(
            # torch.nn.BatchNorm3d(64),
            # torch.nn.LeakyReLU(0.2),
            torch.nn.Conv3d(64, 64, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            torch.nn.Conv3d(64, 128, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            torch.nn.Conv3d(128, 256, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            torch.nn.Conv3d(256, 512, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(512),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(512*(w//16)**3, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.n_cls)
        )

        # pc = o3d.PointCloud()
        # pc.points = o3d.Vector3dVector(self.grid_points)
        # o3d.draw_geometries([pc], window_name="grid", width=1000, height=800)

    def forward(self, x):
        # x: batch_size x n x 6

        # batch_size x n x www
        x_to_grid_dis = square_distance(
            x[:, :, :3], self.grid_points.unsqueeze(0).expand_as(
                torch.empty(x.shape[0], self.grid_points.shape[0], self.grid_points.shape[1])
            ).to(x.device))
        # batch_size x n, 每个点找与自己最近的格点
        grid_idx = x_to_grid_dis.topk(k=1, dim=2, largest=False, sorted=False)[1].squeeze(2)
        # 邻接矩阵, 大小为batch_size x www x n
        adjacency_mat = set_col(
            torch.zeros(x.shape[0]*x.shape[1], self.grid_points.shape[0]).to(x.device),
            grid_idx.view(-1), 1
        ).view(x.shape[0], x.shape[1], self.grid_points.shape[0]).permute([0, 2, 1])

        # # 检查下邻接矩阵求对了没
        # for i in range(adjacency_mat.shape[0]):
        #     grid_pc = o3d.PointCloud()
        #     grids = self.grid_points[adjacency_mat[i].sum(1) > 0].numpy()
        #     grid_pc.points = o3d.Vector3dVector(grids)
        #     o3d.draw_geometries([grid_pc], window_name="pc grid", width=1000, height=800)

        adjacency_mat_w = adjacency_mat.sum(dim=2, keepdim=True)
        adjacency_mat_w[adjacency_mat_w == 0] = 1
        adjacency_mat = adjacency_mat / adjacency_mat_w

        # batch_size x C x w x w x w
        point2voxel = torch.matmul(adjacency_mat, self.point2vec(x[:, :, :].permute([0, 2, 1])).permute([0, 2, 1])).permute([0, 2, 1]).view(x.shape[0], -1, self.w, self.w, self.w)
        # print(point2voxel.shape)
        # 3D convolution
        voxel_w_16 = self.conv1(point2voxel)
        voxel_w_8 = self.conv2(voxel_w_16)
        voxel_w_4 = self.conv3(voxel_w_8)
        voxel_w_2 = self.conv4(voxel_w_4)
        # print(voxel_w_2.shape)
        return self.fc(voxel_w_2.view(x.shape[0], -1))


def set_col(x, ind, value):
    h, w = x.shape
    x.view(-1)[(ind+torch.arange(0, h).to(ind.device)*w).view(-1)] = value
    return x.view(h, w)


if __name__ == '__main__':
    # x = torch.Tensor([[0, 0, 0, 0, 0],
    #                   [0, 0, 0, 0, 0]])
    # ind = torch.LongTensor([1, 3])
    # # x[:, ind] = 1
    # # print(x)
    # x = set_col(x, ind, 1)
    # print(x)
    net = PVPNet(w=32, in_channel=6).to(gpu)
    modelnet = ModelNet40(dataset_path="E:/modelnet40_ply_hdf5_2048")
    modelnet_loader = data.DataLoader(dataset=modelnet, batch_size=2, shuffle=False)
    for points, labels in modelnet_loader:
        points, labels = points.to(gpu), labels.to(gpu)
        x = net(points)
        print(x.shape)
        break