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

import random
import cv2
from tqdm import tqdm

# from lcapt.analysis import make_feature_grid
# from lcapt.lca import LCAConv2D
# from lcapt.metric import compute_l1_sparsity, compute_l2_error

from sklearn.metrics import f1_score, accuracy_score, recall_score

from fly_master.src.models.attention.sblocal_attention import SBLocalAttention

class AccidentDataset(Dataset):
    def __init__(self, dataset_path, video_list, data_info, split_set, clip_frames, transform=None):
        self.dataset_path = dataset_path
        self.video_list = video_list
        self.transform = transform
        self.video_labels = data_info[split_set]  # still a key
        self.clip_frames = clip_frames

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, item):
        # get video ID
        video_id = self.video_list[item]
        data_label = self.video_labels[video_id]
        # retrieve the whole video data
        path_for_video = self.dataset_path + video_id

        # select the first self.clip_frames frames
        trans_image_list = []
        for img in range(self.clip_frames):
            img_path = os.path.join(path_for_video, (str(img) + '.jpg'))
            data = Image.open(img_path)
            # image transformation
            if self.transform is not None:
                trans_image_list.append(self.transform(data))
                # shape of data: t(self.clip_frames) x 3 x dim_h x dim_w
        video_tensor = torch.stack(trans_image_list)

        timestamp_list = torch.zeros(self.clip_frames)
        if int(data_label) < self.clip_frames:
            bi_label = torch.ones(1)  # -> represent that there is an accident in this video clip
            timestamp_list[int(data_label)] = 1  # tensor([0., 0., 0., 1., 0.]
        else:  # there are no accident in this video clip
            bi_label = torch.zeros(1)

        return {'data': video_tensor, 'timestamp': timestamp_list, 'label': bi_label}


class SelfAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.features = args.features
        self.h = args.dim_h
        self.w = args.dim_w
        self.n_L = args.clip_frames

        n_input = args.features * args.dim_h * args.dim_w

        n_class = 1

        # Self-Attention层
        self.attention = nn.MultiheadAttention(n_input, num_heads=args.num_heads, dropout=args.dropout_rate)
        # self.attention = SBLocalAttention(local_context=5, dim_heads=args.num_heads, attention_dropout=args.dropout_rate)

        # 线性层
        self.linear = nn.Linear(n_input, n_class)

    def forward(self, sparse_represent):
        # sparse_represent的形状: batch size x self.n_L x 1 x self.features x self.h x self.w
        b_c = sparse_represent.shape[0]
        x = sparse_represent.reshape(b_c, self.n_L, -1)

        # 交换维度以符合self-attention的输入要求
        x = x.permute(1, 0, 2)  # 将batch和sequence length交换位置，使得sequence length成为第一个维度

        # Self-Attention计算
        attn_output, _ = self.attention(x, x, x)

        # 恢复原始维度顺序
        attn_output = attn_output.permute(1, 0, 2)  # 恢复原始顺序

        # 使用线性层进行分类
        z = self.linear(attn_output[:, -1, :])  # 取最后一个时间步的输出进行分类

        return z

def evaluation(sparse_model, mlp_model, criterion, data_loader, data_len, device):
    sparse_model.eval()
    mlp_model.eval()
    pre_layer = nn.Sigmoid()
    Loss = 0
    counter = 0
    for data in tqdm(data_loader):
        video_clips = data['data'].to(device=device)

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
                acc_image = video_clips[video][frame]  # shape: No.features x h x w
                # sparse encoding
                trans_acc_image = torch.stack([acc_image])  # shape: 1 x No.features x h x w
                sparse_encoding = sparse_model(trans_acc_image)[1]  # shape: 1 x No.features x h x w
                inner_list.append(sparse_encoding)  # clip_frames : 1 x No.features x h x w in a list
            input_list.append(torch.stack(inner_list))  # batch size: clip_frames x 1 x No.features x h x w
        input_list = torch.stack(input_list)  # batch size x clip_frames x 1 x No.features x h x w
        predictions = mlp_model(input_list)

        label_prob = (pre_layer(predictions) >= 0.5).int().detach().to(device='cpu').numpy()

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

# remove video with less than 100 frames
def remove_short_video(video_list, data_path):
    more_than_100 = []
    for V_id in video_list:
        # get path
        V_path = data_path + V_id + '/'
        # get frames num
        V_len = len(os.listdir(V_path))
        if V_len >= 100:
            more_than_100.append(V_id)
    return more_than_100


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
    args.train_list = remove_short_video(train_list, args.data_path)  # 001102, 000920, 001268, ...
    args.val_list = remove_short_video(val_list, args.data_path)  # 001102, 000920, 001268, ...
    args.test_list = remove_short_video(test_list, args.data_path)  # 001102, 000920, 001268, ...

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
                                    transform=data_transforms['train'])

    val_dataset = AccidentDataset(args.data_path,
                                  args.val_list,
                                  dataset_sep,
                                  'val',
                                  args.clip_frames,
                                  transform=data_transforms['test'])

    test_dataset = AccidentDataset(args.data_path,
                                   args.test_list,
                                   dataset_sep,
                                   'test',
                                   args.clip_frames,
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
    # total_training_records = 0
    # positive_training_records = 0
    # for data in train_loader:
    #     total_training_records += np.array(data['label']).shape[0]
    #     positive_training_records += np.array(data['label']).sum()
    #
    #
    # total_val_records = 0
    # positive_val_records = 0
    # for data in val_loader:
    #     total_val_records += np.array(data['label']).shape[0]
    #     positive_val_records += np.array(data['label']).sum()
    #
    # total_test_records = 0
    # positive_test_records = 0
    # for data in test_loader:
    #     total_test_records += np.array(data['label']).shape[0]
    #     positive_test_records += np.array(data['label']).sum()

    print('Dataset preparation finished!')
    print('Dataset Info: \n'
          f'No. of Videos in Training Set - {train_dataset.__len__()} \n'
          f'No. of Videos in Validation Set - {val_dataset.__len__()} \n'
          f'No. of Videos in Testing Set - {test_dataset.__len__()}'
          )
    # print(f'Total No. Videos: {train_dataset.__len__() + val_dataset.__len__() + test_dataset.__len__()}')
    # print(f'Positive records ratio in training set: {round(positive_training_records / total_training_records, 2)}')
    # print(f'Positive records ratio in val set: {round(positive_val_records / total_val_records, 2)}')
    # print(f'Positive records ratio in testing set: {round(positive_test_records / total_test_records, 2)}')

    # Load pretrained LCA model
    # and switch it to eval mode
    sparse_coding_lca = torch.load(args.pretrained_lca)
    sparse_coding_lca.to(device=args.device)
    sparse_coding_lca.eval()
    print('***Pretrained LCA model is loaded!***')

    # Build MLP models
    # accept b_s x t(args.clip_frames,) x features x h x w as inputs
    my_mlp = SelfAttention(args).to(device=args.device)

    # optimizer & loss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(my_mlp.parameters(), lr=args.learning_rate)
    print('***Mlp is created!***')

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
        my_mlp.train()
        sparse_coding_lca.eval()
        for data in tqdm(train_loader):
            video_clips = data['data'].to(device=args.device)
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
            for video in range(b_s):
                inner_list = []
                for frame in range(frame_no):
                    acc_image = video_clips[video][frame] # shape: No.features x h x w
                    # sparse encoding
                    trans_acc_image = torch.stack([acc_image]) # shape: 1 x No.features x h x w
                    sparse_encoding = sparse_coding_lca(trans_acc_image)[1]  # shape: 1 x No.features x h x w
                    inner_list.append(sparse_encoding) # args.clip_frames : 1 x No.features x h x w in a list
                input_list.append(torch.stack(inner_list)) # batch size: args.clip_frames x 1 x No.features x h x w
            input_list = torch.stack(input_list) # batch size x args.clip_frames x 1 x No.features x h x w
            predictions = my_mlp(input_list)

            # calculate the loss
            loss = criterion(predictions, frame_labels)

            # backward
            loss.backward()

            # update the parameters
            optimizer.step()

        # print(f'Positive records ratio: {round(positive_training_records / total_training_records, 2)}')

        print(f'****Training of epoch: {epoch + 1} finished!****')

        print('Model performance:')
        # evaluation on the training & evaluation set
        loss_train, f1_train, acc_train, recall_train = evaluation(sparse_coding_lca,
                                                                   my_mlp,
                                                                   criterion,
                                                                   train_loader,
                                                                   train_dataset.__len__(),
                                                                   args.device)
        loss_val, f1_val, acc_val, recall_val = evaluation(sparse_coding_lca,
                                                           my_mlp,
                                                           criterion,
                                                           val_loader,
                                                           val_dataset.__len__(),
                                                           args.device)
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
                                                           best_model,
                                                           criterion,
                                                           test_loader,
                                                           test_dataset.__len__(),
                                                           args.device)
    # save training log
    training_log['loss_test'] = loss_test
    training_log['acc_test'] = acc_test
    training_log['f1_test'] = f1_test
    training_log['recall_test'] = recall_test
    file = open(f'{args.mlp_model_saving_path}/training_log_{args.ct}.p', 'wb')
    pickle.dump(training_log, file)
    file.close()
    print('***All training, validation and testing logs are saved!***')

def test():
    parser = argparse.ArgumentParser(description='Smart Traffic Project - Detect car accident')

    parser.add_argument('--pretrained_lca', type=str, help='Path of the pretrain LCA model')
    parser.add_argument('--mlp_model_saving_path', default='./LCA_with_attention_v1', type=str)

    # data file settings
    # parser.add_argument('--data_path', default=r'/mnt/disk1/uncleaned_data/')
    parser.add_argument('--data_path', default=r'datasets/')
    parser.add_argument('--data_labels', default='aligned_split.p')

    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--clip_frames', default=80, type=int) # how many frames you want to include in a clip
    # dim_h & dim_w should be same with the values that were used in the stage 1 (Sparse coding modelling)
    parser.add_argument('--dim_h', default=32, type=int,
                        help='Dimension of input frame on h direction')
    parser.add_argument('--dim_w', default=48, type=int,
                        help='Dimension of input frame on w direction')
    parser.add_argument('--features', default=32, type=int,
                        help='No. sparsity features')
    parser.add_argument('--num_heads',default=3, type=int)
    parser.add_argument('--dropout_rate', default=0.3)

    # model training settings
    parser.add_argument('--in_chans', default=3, type=int,
                        help='Dimension of input channels, here is RGB=3')
    parser.add_argument('--hidden_size', default=64, type=int,
                        help='No. of nuerons in MLP hidden layer')
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help='learning rate of MLP')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--ct', default='xxx', help='current time')
    parser.add_argument('--SEED', default=2023, type=int, help='random state')
    # training strategy for preventing overfiting
    parser.add_argument('--pat', default=5, type=int, help='patience of early stopping')

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device

    my_mlp = SelfAttention(args).to(device=args.device)
    print("--------------------Successful!----------------------")
    print(my_mlp)

def main():
    parser = argparse.ArgumentParser(description='Smart Traffic Project - Detect car accident')

    parser.add_argument('--pretrained_lca', default='Image_LCA_08_18_2024_23_17_03/SparseCoding_08_18_2024_23_17_03.pt', type=str, help='Path of the pretrain LCA model')
    parser.add_argument('--mlp_model_saving_path', default='./LCA_with_attention_v1', type=str)

    # data file settings
    # parser.add_argument('--data_path', default=r'/mnt/disk1/uncleaned_data/')
    parser.add_argument('--data_path', default=r'datasets/')
    parser.add_argument('--data_labels', default='aligned_split.p')

    parser.add_argument('--batch_size', default=2, type=int, help='batch size')
    parser.add_argument('--clip_frames', default=20, type=int) # how many frames you want to include in a clip
    # dim_h & dim_w should be same with the values that were used in the stage 1 (Sparse coding modelling)
    parser.add_argument('--dim_h', default=32, type=int,
                        help='Dimension of input frame on h direction')
    parser.add_argument('--dim_w', default=36, type=int,
                        help='Dimension of input frame on w direction')
    parser.add_argument('--features', default=32, type=int,
                        help='No. sparsity features')

    parser.add_argument('--num_heads',default=3, type=int)
    parser.add_argument('--dropout_rate', default=0.3)

    # model training settings
    parser.add_argument('--hidden_size', default=64, type=int,
                        help='No. of nuerons in MLP hidden layer')
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help='learning rate of MLP')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--ct', default='xxx', help='current time')
    parser.add_argument('--SEED', default=2023, type=int, help='random state')
    # dim_h & dim_w should be same with the values that were used in the stage 1 (Sparse coding modelling)
    # parser.add_argument('--dim_h', default=128, type=int,
    #                     help='Dimension of input frame on h direction')
    # parser.add_argument('--dim_w', default=96, type=int,
    #                     help='Dimension of input frame on w direction')
    # parser.add_argument('--features', default=64, type=int,
    #                     help='No. sparsity features')

    parser.add_argument('--in_chans', default=3, type=int,
                        help='Dimension of input channels, here is RGB=3')
    # training strategy for preventing overfiting
    parser.add_argument('--pat', default=5, type=int, help='patience of early stopping')

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

if __name__ == '__main__':
    main()
    # test()