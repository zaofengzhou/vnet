from train_def import *


if __name__ == '__main__':
    BATCH_SIZE = 4  # 2

    data_path = []  # 装图所在subset的绝对地址，如 [D:\datasets\sk_output\bbox_image\subset0]
    label_path = []  # 装标签所在subset的绝对地址，与上一行一致，为对应关系
    for i in range(0, 1):  # 0,1,2,3,4,5,6,7   训练集
        data_path.append(bbox_img_path + fengefu + 'subset%d' % i)  # 放入对应的训练集subset的绝对地址
        label_path.append(bbox_msk_path + fengefu + 'subset%d' % i)
    dataset_train = myDataset(data_path, label_path)  # 送入dataset
    print(len(dataset_train))
    train_loader = torch.utils.data.DataLoader(dataset_train,  # 生成dataloader
                                               batch_size=BATCH_SIZE, shuffle=False,
                                               num_workers=0)  # 16)  # 警告页面文件太小时可改为0

    for data, target in train_loader:
        print(data)
        print(target)
        break
