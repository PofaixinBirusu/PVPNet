# PVPNet
参数下载地址：https://pan.baidu.com/s/1zOlLZPs9ox-PKaujNF0DHA  提取码：8gay
### 实验结果
Classification in modelnet40 dataset
| Method | Input | Points | Accuracy |
| :-----| :----- | :----- | :----- |
| PointNet| P | 1024 | 89.2% |
| PointNet++| P | 1024 | 90.7% |
| PointNet++| P, N | 5000 | 91.9% |
| SO-Net| P, N | 2048 | 90.9% |
| KD-Net| P | kd-tree | 91.8% |
| PointConv| P, N | 1024 | 92.5% |
| PointCNN| P | 1024 | 92.5% |
| DGCNN| P | 1024 | <b>92.9%</b> |
| KPConv| P | 6500 | <b>92.9%</b> |
| <b>PVPNet (ous)</b>| P | 2048 | 92.3% |
| <b>PVPNet (ous)</b>| P, N | 2048 | <b>92.9%</b> |

Part segmentation results on ShapeNet part dataset. Metric is mIoU(%) on points
|  | mean | airplane | bag | cap | car | chair | ear-phone | guitar | knife | lamp | laptop | mortor | mug | pistol | rocket | stake-board | table | winning categories |
| :-----| :----- | :----- | :----- | :----- | :----- | :----- | :----- | :----- | :----- | :----- | :----- | :----- | :----- | :----- | :----- | :----- | :----- | :----- |
| shapes |  | 2690 |76 |55 |898 |3758 |69 |787 |392 |1547 |451 |202 |184 |283 |66 |152 |5271 |  |
| PointNet | 83.7 | 83.4 |78.7 |82.5 |74.9 |89.6 |73.0 |<b>91.5</b> |85.9 |80.8 |95.3 |65.2 |93.0 |81.2 |57.9 |72.8 |80.6 | 1 |
| PointNet++ | 85.1 | 82.4 |79.0 |87.7 |77.3 |90.8 |71.8 |91.0 |85.9 |83.7 |95.3 |<b>71.6</b> |94.1 |81.3 |58.7 |<b>76.4</b> |82.6 | 2 |
| KD-Net | 82.3 | 80.1 |74.6 |74.3 |70.3 |88.6 |73.5 |90.2 |87.2 |81.0 |94.9 |57.4 |86.7 |78.1 |51.8 |69.9 |80.3 | 0 |
| LocalFeatureNet | 84.3 | <b>86.1</b> |73.0 |54.9 |<b>77.4</b> |88.8 |55.0 |90.6 |86.5 |75.2 |<b>96.1</b> |57.3 |91.7 |<b>83.1</b> |53.9 |72.5 |<b>83.8</b> | 4 |
| DGCNN | 85.1 | 84.2 |<b>83.7</b> |84.4 |77.1 |90.9 |<b>78.5</b> |<b>91.5</b> |87.3 |82.9 |96.0 |67.8 |93.3 |82.6 |<b>59.7</b> |75.5 |82.0 | 4 |
| <b>PVPNet (ours)</b> | <b>85.4</b> | 83.0 |81.9 |<b>90.9</b> |<b>77.4</b> |<b>91.5</b> |72.5 |89.7 |<b>88.4</b> |<b>85.9</b> |95.8 |61.0 |93.6 |81.9 |55.1 |71.2 |82.8 | <b>6</b> |

Completion method

![1](https://github.com/PofaixinBirusu/PVPNet/blob/main/images/PVPNet-Completion.png)
