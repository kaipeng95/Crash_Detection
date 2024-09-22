import argparse
from datetime import datetime
import pickle
import numpy as np
import os

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.functional

import random
from flow_model.utils import flow_viz
import cv2
from tqdm import tqdm

# from lcapt.analysis import make_feature_grid
# from lcapt.lca import LCAConv2D
# from lcapt.metric import compute_l1_sparsity, compute_l2_error

from sklearn.metrics import f1_score, accuracy_score, recall_score
from flow_model.raft import RAFT

class AccidentDataset(Dataset):
    def __init__(self, dataset_path, video_list, data_info, split_set, clip_frames,  h, w, transform=None):
        self.dataset_path = dataset_path
        self.video_list = video_list
        self.transform = transform
        self.video_labels = data_info[split_set]  # still a key
        self.clip_frames = clip_frames
        self.dim_h = h
        self.dim_w = w

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, item):
        # get video ID
        video_id = self.video_list[item]
        data_label = self.video_labels[video_id]
        # retrieve the whole video data
        path_for_video = self.dataset_path + video_id

        # 为flow设计的transform
        data_transforms_flow = transforms.Compose([
                #         transforms.RandomRotation(5),
                transforms.Resize((self.dim_h, self.dim_w)),
                #         transforms.RandomHorizontalFlip(),
                #         transforms.CenterCrop(224),
                transforms.PILToTensor(),
                # transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        # select the first self.clip_frames frames
        trans_image_list = []
        no_trans_image_list = []
        for img in range(self.clip_frames):
            img_path = os.path.join(path_for_video, (str(img) + '.jpg'))
            data = Image.open(img_path)
            # image transformation
            if self.transform is not None:
                trans_image_list.append(self.transform(data))
                no_trans_image_list.append(data_transforms_flow(data).float())
                # shape of data: t(self.clip_frames) x 3 x dim_h x dim_w
        video_tensor = torch.stack(trans_image_list)
        video_tensor_notrans = torch.stack(no_trans_image_list)

        timestamp_list = torch.zeros(self.clip_frames)
        if int(data_label) < self.clip_frames:
            bi_label = torch.ones(1)  # -> represent that there is an accident in this video clip
            timestamp_list[int(data_label)] = 1  # tensor([0., 0., 0., 1., 0.]
        else:  # there are no accident in this video clip
            bi_label = torch.zeros(1)

        return {'data': [video_tensor, video_tensor_notrans], 'timestamp': timestamp_list, 'label': bi_label}

# remove video with less than 100 frames
def remove_short_video(video_list, data_path, clip_frames):
    more_than_clip_frames = []
    for V_id in video_list:
        # get path
        V_path = data_path + V_id + '/'
        # get frames num
        V_len = len(os.listdir(V_path))
        # if V_len >= 100:
        if V_len >= clip_frames:
            more_than_clip_frames.append(V_id)
    return more_than_clip_frames

# 光流可视化
def viz(img, flo):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2, 1, 0]] / 255.0)
    cv2.waitKey()

# MHSA模型build
class SelfAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.features = args.features-2
        self.h = int(args.dim_h/8)
        self.w = int(args.dim_w/8)
        self.n_L = int(args.clip_frames/args.frame_extra_rate)

        n_input = self.features * self.h * self.w
        n_class = 1

        self.conv1 = nn.Conv2d(in_channels=args.features, out_channels=args.features, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=args.features, out_channels=args.features-2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=args.features-2, out_channels=args.features-2, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.BatchNorm2d(num_features=args.features)
        self.norm2 = nn.BatchNorm2d(num_features=args.features-2)
        self.norm3 = nn.BatchNorm2d(num_features=args.features-2)
        self.relu = nn.ReLU()

        # Self-Attention层
        self.attention = nn.MultiheadAttention(n_input, num_heads=args.num_heads, dropout=args.dropout_rate)
        self.norm4 = nn.BatchNorm1d(num_features=n_input)

        # 线性层
        self.linear = nn.Linear(n_input, n_class)
        # 激活函数
        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, sparse_represent):
        # 输入 x 的形状为 (batch_size, n_L, 1, features, h, w)
        batch_size, n_L, _, features, h, w = sparse_represent.shape
        # 调整形状为 (batch_size * n_L, features, h, w)
        sparse_represent = sparse_represent.view(batch_size * n_L, features, h, w)
        # 进行二维卷积x3
        # sparse_represent = self.conv3(self.conv2(self.conv1(sparse_represent)))
        sparse_represent = self.norm3(self.conv3(self.norm2(self.conv2(self.norm1(self.conv1(sparse_represent))))))
        # sparse_represent = self.relu(self.norm1(self.conv1(sparse_represent)))
        # sparse_represent = self.relu(self.norm2(self.conv2(sparse_represent)))
        # sparse_represent = self.relu(self.norm3(self.conv3(sparse_represent)))
        # 卷积后调整形状为 (batch_size, n_L, features, h/8, w/8)
        h_out, w_out = sparse_represent.shape[2], sparse_represent.shape[3]
        sparse_represent = sparse_represent.view(batch_size, n_L, features - 2, h_out, w_out)
        # sparse_represent的形状: batch size x self.n_L x 1 x self.features x self.h x self.w
        # print(sparse_represent[0, :, :, :, :])
        # b_c = sparse_represent.shape[0]
        x = sparse_represent.reshape(batch_size, self.n_L, -1)

        # 交换维度以符合self-attention的输入要求
        print(f'x[0, -1, :]:\n{x[0, -1, :]}')
        x = x.permute(1, 0, 2)  # 将batch和sequence length交换位置，使得sequence length成为第一个维度
        # Self-Attention计算
        attn_output, _ = self.attention(x, x, x)
        # 恢复原始维度顺序
        attn_output = attn_output.permute(1, 0, 2)  # 恢复原始顺序
        # print(attn_output.shape)
        # print(attn_output[:, -1, :].shape)
        attn_output = attn_output.permute(0, 2, 1)
        attn_output = self.norm4(attn_output)
        attn_output = attn_output.permute(0, 2, 1)
        print(f'attn_output[0, -1, :]:\n{attn_output[0, -1, :]}')

        # 使用线性层进行分类
        z = self.linear(attn_output[:, -1, :])
        print(z)
        z = self.sigmoid(z) # 取最后一个时间步的输出进行分类
        # z = self.act(self.linear(attn_output[:, -1, :])) # 取最后一个时间步的输出进行分类


        return z

def evaluation(sparse_model, flow_net, mlp_model, criterion, data_loader, data_len, device, extra_rate):
    sparse_model.eval()
    flow_net.eval()
    mlp_model.eval()
    pre_layer = nn.Sigmoid()
    frame_extra_rate = extra_rate
    Loss = 0
    counter = 0
    for data in tqdm(data_loader):
        video_clips = data['data'][0].to(device=device)
        video_clips_flow = data['data'][1].to(device=device)

        if counter == 0:
            y_true = data['label'].numpy()  # shape: batch_size x 1.
        else:
            y_true = np.concatenate((y_true, data['label'].numpy()), axis=0)

        # 实际上是每个video的label
        frame_labels = data['label'].to(device=device)  # shape: batch_size x 1.

        # forward
        b_s = video_clips.shape[0]
        input_list = []
        frame_no = video_clips.shape[1]
        for video in range(b_s):
            inner_list = []
            for frame in range(frame_no):
                # 抽帧
                if frame % frame_extra_rate == 0:
                    acc_image1 = video_clips[video][frame]  # shape: No.features x h x w
                    trans_acc_image1 = torch.stack([acc_image1])  # shape: 1 x No.features x h x w

                    acc_image1_flow = video_clips_flow[video][frame]  # shape: No.features x h x w
                    # 考虑到相邻帧外观表示相差不大，于是选择与间隔frame_extra_rate帧获取光流
                    acc_image2_flow = video_clips_flow[video][
                        frame + frame_extra_rate] if frame != frame_no - frame_extra_rate else acc_image1_flow
                    trans_acc_image1_flow = torch.stack([acc_image1_flow])  # shape: 1 x No.features x h x w
                    trans_acc_image2_flow = torch.stack([acc_image2_flow])  # shape: 1 x No.features x h x w

                    # flow encoding
                    flow_low, flow_encoding = flow_net(trans_acc_image1_flow, trans_acc_image2_flow, iters=20,
                                                       test_mode=True)  # shape: 1 x 2(dim) x h x w
                    # sparse encoding
                    sparse_encoding = sparse_model(trans_acc_image1)[1]  # shape: 1 x No.features x h x w
                    full_encoding = torch.concatenate((sparse_encoding, flow_encoding), dim=1)
                    inner_list.append(full_encoding)
            input_list.append(torch.stack(inner_list))  # batch size: clip_frames x 1 x No.features x h x w
        input_list = torch.stack(input_list)  # batch size x clip_frames x 1 x No.features x h x w

        predictions = mlp_model(input_list)

        # label_prob = (pre_layer(predictions) >= 0.5).int().detach().to(device='cpu').numpy()
        label_prob = (predictions >= 0.5).int().detach().to(device='cpu').numpy()

        # calculate the loss
        Loss += (criterion(predictions, frame_labels) * b_s).item()

        if counter == 0:
            y_pred = label_prob  # shape: batch_size x 1.
        else:
            y_pred = np.concatenate((y_pred, label_prob), axis=0)

        counter += 1
    # calculate precision, recall, F1 score
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return Loss / data_len, f1, accuracy, recall


def train(args):
    # clean dataset -> select data with more than 100 frames
    # same strategy that we used in the first stage
    # load dataset pickles file
    pickle_file_path = args.data_labels # aligned_split.p
    dataset_sep = pickle.load(open(pickle_file_path, "rb"))
    # train set
    train_list = dataset_sep['train'].keys()  # 001102, 000920, 001268, ...
    # val set
    val_list = dataset_sep['val'].keys()  # 001102, 000920, 001268, ...
    # test set
    test_list = dataset_sep['test'].keys()  # 001102, 000920, 001268, ...

    # clean dataset
    args.train_list = remove_short_video(train_list, args.data_path, args.clip_frames)  # 001102, 000920, 001268, ...
    args.val_list = remove_short_video(val_list, args.data_path, args.clip_frames)  # 001102, 000920, 001268, ...
    args.test_list = remove_short_video(test_list, args.data_path, args.clip_frames)  # 001102, 000920, 001268, ...

    data_transforms = {
        'train': transforms.Compose([
            #         transforms.RandomRotation(5),
            transforms.Resize((args.dim_h, args.dim_w)),
            #         transforms.RandomHorizontalFlip(),
            #         transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((args.dim_h, args.dim_w)),
            #         transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # make dataset
    # all outputs from the accident dataset are tensors
    train_dataset = AccidentDataset(args.data_path,
                                    args.train_list,
                                    dataset_sep,
                                    'train',
                                    args.clip_frames,
                                    args.dim_h,
                                    args.dim_w,
                                    transform=data_transforms['train'],)

    val_dataset = AccidentDataset(args.data_path,
                                  args.val_list,
                                  dataset_sep,
                                  'val',
                                  args.clip_frames,
                                  args.dim_h,
                                  args.dim_w,
                                  transform=data_transforms['test'])

    test_dataset = AccidentDataset(args.data_path,
                                   args.test_list,
                                   dataset_sep,
                                   'test',
                                   args.clip_frames,
                                   args.dim_h,
                                   args.dim_w,
                                   transform=data_transforms['test'])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    '''
    total_training_records = 0
    positive_training_records = 0
    for data in train_loader:
        total_training_records += np.array(data['label']).shape[0]
        positive_training_records += np.array(data['label']).sum()


    total_val_records = 0
    positive_val_records = 0
    for data in val_loader:
        total_val_records += np.array(data['label']).shape[0]
        positive_val_records += np.array(data['label']).sum()

    total_test_records = 0
    positive_test_records = 0
    for data in test_loader:
        total_test_records += np.array(data['label']).shape[0]
        positive_test_records += np.array(data['label']).sum()
    '''
    print('Dataset preparation finished!')
    print('Dataset Info: \n'
          f'No. of Videos in Training Set - {train_dataset.__len__()} \n'
          f'No. of Videos in Validation Set - {val_dataset.__len__()} \n'
          f'No. of Videos in Testing Set - {test_dataset.__len__()}'
          )
    print(f'Total No. Videos: {train_dataset.__len__() + val_dataset.__len__() + test_dataset.__len__()}')
    '''
    print(f'Positive records ratio in training set: {round(positive_training_records / total_training_records, 2)}')
    print(f'Positive records ratio in val set: {round(positive_val_records / total_val_records, 2)}')
    print(f'Positive records ratio in testing set: {round(positive_test_records / total_test_records, 2)}')
    '''
    # Load pretrained LCA model
    # and switch it to eval mode
    sparse_coding_lca = torch.load(args.pretrained_lca)
    sparse_coding_lca.to(device=args.device)
    sparse_coding_lca.eval()
    print('***Pretrained LCA model is loaded!***')

    # Load pretrained FlowNet model
    # and switch it to eval mode
    flow_net = torch.nn.DataParallel(RAFT(args))
    flow_net.load_state_dict(torch.load(args.pretrained_flow))
    flow_net = flow_net.module
    flow_net.to(device=args.device)
    flow_net.eval()
    print('***Pretrained FlowNet model is loaded!***')


    # Build MLP models
    # accept b_s x t(args.clip_frames,) x features x h x w as inputs
    my_mlp = SelfAttention(args).to(device=args.device)

    # optimizer & loss
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(my_mlp.parameters(), lr=args.learning_rate)
    print('***Mlp is created!***')

    # 遍历模型的参数并计算总数
    total_params = 0
    for params in my_mlp.parameters():
        total_params += params.numel()
    print(f"Total number of my_mlp model parameters: {total_params/1e6:.2f}M")

    patience = args.pat
    early_stopping_counter = 0

    # var used to save training log
    log_epoch_idx = []

    train_loss_log = []
    train_acc_log = []
    train_f1_log = []
    train_recall_log = []

    vali_loss_log = []
    vali_acc_log = []
    vali_f1_log = []
    vali_recall_log = []

    training_log = {}

    # calculate No. positive & negative records
    # total_training_records = 0
    # positive_training_records = 0

    print('***Start training!***')

    for epoch in range(args.epochs):
        sparse_coding_lca.eval()
        flow_net.eval()
        my_mlp.train()
        for data in tqdm(train_loader):
            video_clips = data['data'][0].to(device=args.device)
            video_clips_flow = data['data'][1].to(device=args.device)

            # total_training_records += np.array(data['label']).shape[0]
            # positive_training_records += np.array(data['label']).sum()

            # 实际上是每个video的label
            frame_labels = data['label'].to(device=args.device)  # shape: batch_size x 1.

            # clean the old gradient info
            optimizer.zero_grad()

            # forward
            b_s = video_clips.shape[0]
            frame_no = video_clips.shape[1] # args.clip_frames

            input_list = []
            # input_list = torch.zeros((0, frame_no,args.features,args.dim_h,args.dim_w)).to(device=args.device)  # [batch_size,clip_frames,feature,h,w]
            for video in range(b_s):
                inner_list = []
                # input_list_tmp = torch.zeros((0, args.features,args.dim_h,args.dim_w)).to(device=args.device)
                for frame in range(frame_no):
                    # 抽帧
                    if frame % args.frame_extra_rate == 0:
                        acc_image1 = video_clips[video][frame] # shape: No.features x h x w
                        trans_acc_image1 = torch.stack([acc_image1])  # shape: 1 x No.features x h x w

                        acc_image1_flow = video_clips_flow[video][frame] # shape: No.features x h x w
                        # 考虑到相邻帧外观表示相差不大，于是选择与间隔frame_extra_rate帧获取光流
                        acc_image2_flow = video_clips_flow[video][frame+args.frame_extra_rate] if frame != frame_no-args.frame_extra_rate else acc_image1_flow
                        trans_acc_image1_flow = torch.stack([acc_image1_flow])  # shape: 1 x No.features x h x w
                        trans_acc_image2_flow = torch.stack([acc_image2_flow])  # shape: 1 x No.features x h x w

                        # flow encoding
                        flow_low, flow_encoding = flow_net(trans_acc_image1_flow, trans_acc_image2_flow, iters=20, test_mode=True)  # shape: 1 x 2(dim) x h x w
                        # 可视化光流效果
                        # viz(trans_acc_image1_flow, flow_encoding.detach())
                        # 打印光流信息
                        # print(f'flow_encoding.shape:{flow_encoding.shape}\nflow_encoding:{flow_encoding}')

                        # sparse encoding
                        sparse_encoding = sparse_coding_lca(trans_acc_image1)[1]  # shape: 1 x No.features x h x w
                        full_encoding = torch.concatenate((sparse_encoding, flow_encoding), dim=1)

                    # input_list_tmp = torch.cat([input_list_tmp, (torch.concatenate((sparse_encoding, flow_encoding), dim=1))], dim=0)
                # input_list = torch.cat([input_list, input_list_tmp.unsqueeze(dim=0)], dim=0)

                    # print("No.{} inner_list:{}".format(frame, input_list_tmp.shape))
                #     if frame == 0:
                #         inner_list = torch.concatenate((sparse_encoding, flow_encoding), dim=1)
                #         print("No.{} flow + sparse:{}".format(frame + 1, inner_list.shape))
                #     else:
                #         full_encoding = torch.concatenate((sparse_encoding, flow_encoding), dim=1)
                #         print("No.{} flow + sparse:{}".format(frame + 1, full_encoding.shape))
                #         inner_list = torch.cat([inner_list, full_encoding], dim=0)
                #     print("No.{} inner_list{}".format(frame + 1, inner_list.shape))   # clip_frames x No.features x h x w
                # if video == 0:
                #     input_list = inner_list.unsqueeze(dim=0)
                # else:
                #     input_list = torch.cat([input_list, inner_list.unsqueeze(dim=0)], dim=0) # batch_size x clip_frames x No.features x h x w
                # print("input_list[video]:{}".format(input_list[video].shape))
                # print("第 %d 个batch完成！" % video)

                        inner_list.append(full_encoding) # args.clip_frames : 1 x No.features x h x w in a list
                input_list.append(torch.stack(inner_list)) # batch size: args.clip_frames x 1 x No.features x h x w
            input_list = torch.stack(input_list) # batch size x args.clip_frames x 1 x No.features x h x w
            # print("input_list:{}".format(input_list.shape))

            '''
            # 抽帧
            print("开始抽帧......")
            atten_input = []
            # for bs in range(b_s):
            for bs in tqdm(range(b_s)):
                tmp_data = []
                for frame in range(frame_no):
                    if frame % 5 == 0:
                        tmp_data.append(input_list[bs][frame])
                atten_input.append(torch.stack(tmp_data))
            atten_input = torch.stack(atten_input)
            print("atten_input:{}".format(atten_input.shape))
            print("<------------------------------------------------->")
            '''
            predictions = my_mlp(input_list)
            # print("predictions shape: {}".format(predictions.shape))
            # print(predictions)

            # print("frame_labels shape: {}".format(frame_labels.shape))
            # print(frame_labels)

            # calculate the loss
            loss = criterion(predictions, frame_labels)
            print(f'predictions:{predictions}')
            print(f'frame_labels:{frame_labels}')
            print(f'Loss item:{loss:.2f}')
            print("<-------------------------->")

            # backward
            loss.backward()

            # update the parameters
            optimizer.step()

        # print(f'Positive records ratio: {round(positive_training_records / total_training_records, 2)}')

        print(f'****Training of epoch: {epoch + 1} finished!****')

        print('Model performance:')
        # evaluation on the training & evaluation set
        loss_train, f1_train, acc_train, recall_train = evaluation(sparse_coding_lca,
                                                                   flow_net,
                                                                   my_mlp,
                                                                   criterion,
                                                                   train_loader,
                                                                   train_dataset.__len__(),
                                                                   args.device,
                                                                   args.frame_extra_rate)
        loss_val, f1_val, acc_val, recall_val = evaluation(sparse_coding_lca,
                                                           flow_net,
                                                           my_mlp,
                                                           criterion,
                                                           val_loader,
                                                           val_dataset.__len__(),
                                                           args.device,
                                                           args.frame_extra_rate)
        # report
        print(f'Training Loss:{round(loss_train, 2)}; Training Acc:{round(acc_train, 2)}; '
              f'Training F1:{round(f1_train, 2)}; Training recall:{round(recall_train, 2)}')
        print('    ***    ')
        print(f'Validation Loss:{round(loss_val, 2)}; Validation Acc:{round(acc_val, 2)}; '
              f'Validation F1:{round(f1_val, 2)}; Validation recall:{round(recall_val, 2)}')

        # save training log
        log_epoch_idx.append(epoch + 1)
        train_loss_log.append(loss_train)
        train_acc_log.append(acc_train)
        train_f1_log.append(f1_train)
        train_recall_log.append(recall_train)

        vali_loss_log.append(loss_val)
        vali_acc_log.append(acc_val)
        vali_f1_log.append(f1_val)
        vali_recall_log.append(recall_val)

        if epoch == 0:
            best_vali_loss = vali_loss_log[-1]
            # reset counter
            early_stopping_counter = 0
            print('***Model parameters saved!***')
            torch.save(my_mlp, f'{args.mlp_model_saving_path}/Attention{args.ct}.pt')
        elif epoch >= 1 and vali_loss_log[-1] <= best_vali_loss:
            # save model
            print('***Model parameters saved!***')
            torch.save(my_mlp, f'{args.mlp_model_saving_path}/Attention{args.ct}.pt')
            # save the best vali loss
            best_vali_loss = vali_loss_log[-1]
            # reset counter
            early_stopping_counter = 0
        elif epoch >= 1 and vali_loss_log[-1] > best_vali_loss:
            print('Validation Loss is increasing!')
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early Stopping! Training stopped.")
                break

    print('***Training finished!***')
    training_log['log_epoch_idx'] = log_epoch_idx
    training_log['train_loss_log'] = train_loss_log
    training_log['train_acc_log'] = train_acc_log
    training_log['train_f1_log'] = train_f1_log
    training_log['train_recall_log'] = train_recall_log

    training_log['vali_loss_log'] = vali_loss_log
    training_log['vali_acc_log'] = vali_acc_log
    training_log['vali_f1_log'] = vali_f1_log
    training_log['vali_recall_log'] = vali_recall_log

    # print(f'Positive records ratio: {round(positive_training_records / total_training_records, 2)}')

    print('***Start testing!***')

    # load best model
    best_model = torch.load(f'{args.mlp_model_saving_path}/Attention{args.ct}.pt')
    best_model.to(device=args.device)
    best_model.eval()
    loss_test, f1_test, acc_test, recall_test = evaluation(sparse_coding_lca,
                                                           flow_net,
                                                           best_model,
                                                           criterion,
                                                           test_loader,
                                                           test_dataset.__len__(),
                                                           args.device,
                                                           args.frame_extra_rate)
    # save training log
    training_log['loss_test'] = loss_test
    training_log['acc_test'] = acc_test
    training_log['f1_test'] = f1_test
    training_log['recall_test'] = recall_test
    file = open(f'{args.mlp_model_saving_path}/training_log_{args.ct}.p', 'wb')
    pickle.dump(training_log, file)
    file.close()
    print('***All training, validation and testing logs are saved!***')


def main():
    parser = argparse.ArgumentParser(description='Smart Traffic Project - Detect car accident')

    parser.add_argument('--pretrained_lca', default='Image_LCA_08_18_2024_23_17_03/SparseCoding_08_18_2024_23_17_03.pt', type=str, help='Path of the pretrain LCA model')
    parser.add_argument('--mlp_model_saving_path', default='./LCA_with_attention_v1', type=str)

    # data file settings
    parser.add_argument('--data_path', default=r'datasets/')
    parser.add_argument('--data_labels', default='aligned_split.p')

    # model training settings
    parser.add_argument('--batch_size', default=2, type=int, help='batch size')

    parser.add_argument('--frame_extra_rate', default=4, type=int, help='Extract one frame every few frames')
    # test
    # parser.add_argument('--frame_extra_rate', default=12, type=int, help='Extract one frame every few frames')
    parser.add_argument('--clip_frames', default=120, type=int) # how many frames you want to include in a clip
    parser.add_argument('--dim_h', default=128, type=int,
                        help='Dimension of input frame on h direction')
    parser.add_argument('--dim_w', default=144, type=int,
                        help='Dimension of input frame on w direction')
    parser.add_argument('--features', default=34, type=int,
                        help='No. sparsity features')
    # dim_h & dim_w should be same with the values that were used in the stage 1 (Sparse coding modelling)
    # parser.add_argument('--dim_h', default=128, type=int,
    #                     help='Dimension of input frame on h direction')
    # parser.add_argument('--dim_w', default=96, type=int,
    #                     help='Dimension of input frame on w direction')
    # parser.add_argument('--dim_h', default=32, type=int,
    #                     help='Dimension of input frame on h direction')
    # parser.add_argument('--dim_w', default=36, type=int,
    #                     help='Dimension of input frame on w direction')
    # parser.add_argument('--features', default=64, type=int,
    #                     help='No. sparsity features')

    parser.add_argument('--hidden_size', default=64, type=int,
                        help='No. of nuerons in MLP hidden layer')
    parser.add_argument('--learning_rate', default=0.001, type=float,
    # parser.add_argument('--learning_rate', default=0.05, type=float,
                        help='learning rate of MLP')

    # parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    # test
    parser.add_argument('--epochs', default=2, type=int, help='number of total epochs to run')
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--ct', default='xxx', help='current time')
    parser.add_argument('--SEED', default=2023, type=int, help='random state')

    parser.add_argument('--in_chans', default=3, type=int,
                        help='Dimension of input channels, here is RGB=3')

    parser.add_argument('--num_heads',default=3, type=int)
    # training strategy for preventing overfiting
    parser.add_argument('--pat', default=5, type=int, help='patience of early stopping')
    # parser.add_argument('--dropout_rate', default=0.3)
    parser.add_argument('--dropout_rate', default=0)

    # RAFT Optical Flow
    parser.add_argument('--pretrained_flow', default='flow_model/checkpoints/raft-kitti.pth', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device
    now = datetime.now()
    current_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    args.ct = current_time
    # args.mlp_model_saving_path = './LCA_with_attention_v1' + current_time

    # fix random state
    random.seed(args.SEED)
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed(args.SEED)

    if not os.path.exists(args.mlp_model_saving_path):
        os.makedirs(args.mlp_model_saving_path)

    train(args)


    #**********************************************
    # test_model = SelfAttention(args).to('cuda')
    # print(test_model)

if __name__ == '__main__':
    main()
