'''
original data: CCTV video data;
label: the frame which denotes the start of an accident is labelled with 1, other frames are all labelled with 0

Data splitting strategy:
There're more than 1K CCTV videos, so we split this whole dataset into training, validation and testing set.
3/5 are assigned to training set and 1/5 and 1/5 are assigned to validation and testing set separately.

In general, the whole traffic accident detection work can be splited into 2 parts:
    1. feature extraction:
        we build an Image-based sparse coding model to learn the data and used it to extract features.
    2. accident detection:
        we build an accident detection model using the features extracted from the well-trained sparse-coding model

Evaluation matrix:
Accuracy
F-1 score
Recall

Novelty:
    1. In order to have better results, we plan to use sparse coding for its better explanation.
    2. We employ the attention mechanism to deal with the tempory information.

'''

import argparse
from datetime import datetime
import random
from tqdm import tqdm
import pickle
import os

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from lcapt.analysis import make_feature_grid
from lcapt.lca import LCAConv2D
from lcapt.metric import compute_l1_sparsity, compute_l2_error

class AccidentDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform

    def __len__(self):
        return len(self.dataset_path)

    def __getitem__(self, item):
        img_path = self.dataset_path[item]
        data = Image.open(img_path)
        # image transformation
        if self.transform is not None:
            data = self.transform(data)
        return data

def Dataset_making(args,split_set,clip_frames):
    # get set and make a list

    # do some changes
    if split_set == 'train':
        video_list = args.train_list
    elif split_set == 'val':
        video_list = args.val_list
    else:
        video_list = args.test_list

    # only save the first clip_frames frames
    image_list = []
    for video in video_list:
        video_path = os.path.join(args.data_path, video)
        # img_len = len(os.listdir(video_path))
        for img in range(clip_frames):
            img_name = str(img) + '.jpg'
            image_list.append(os.path.join(video_path, img_name))
    return image_list

def evaluation(model,dataloader,args,datalen):
    model.eval()
    L2_total_recon = 0
    L1_total_spar = 0
    for batch_num, images in enumerate(tqdm(dataloader)):

        images = images.to(device=args.device)
        inputs, code, recon, recon_error = model(images)
        num_records = len(images)

        # L2 Recon
        # compute_l2_error -> average
        L2_total_recon = L2_total_recon + (compute_l2_error(inputs, recon) * num_records).item()
        # L1 Sparsity
        L1_total_spar = L1_total_spar + (compute_l1_sparsity(code, model.lambda_) * num_records).item()
    # report performance on training data
    L2_train_avg = L2_total_recon / datalen
    L1_train_avg = L1_total_spar / datalen

    return L2_train_avg, L1_train_avg

def train(args):
    # Dataset preparation
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
    training_img_list = Dataset_making(args,'train',args.clip_frames)
    validation_img_list = Dataset_making(args,'val',args.clip_frames)
    testing_img_list = Dataset_making(args,'test',args.clip_frames)
    train_dataset = AccidentDataset(dataset_path=training_img_list, transform=data_transforms['train'])
    vali_dataset = AccidentDataset(dataset_path=validation_img_list, transform=data_transforms['test'])
    test_dataset = AccidentDataset(dataset_path=testing_img_list, transform=data_transforms['test'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    vali_loader = DataLoader(
        vali_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    print('Dataset preparation finished!')
    print('Dataset Info: \n'
          f'No. of Img in Training Set - {train_dataset.__len__()} \n'
          f'No. of Img in Validation Set - {vali_dataset.__len__()} \n'
          f'No. of Img in Testing Set - {test_dataset.__len__()}'
          )

    # Build LCA sparse coding model
    lca = LCAConv2D(
        out_neurons=args.features,
        in_neurons=args.in_chans,
        result_dir=args.lca_model_saving_path,
        kernel_size=args.kernel_size,
        stride=args.STRIDE,
        lambda_=args.LAMBDA,
        tau=args.TAU,
        track_metrics=args.track_metrics,
        eta=args.learning_rate,
        return_vars=['inputs', 'acts', 'recons', 'recon_errors'],
    )
    lca = lca.to(device=args.device)
    print('LCA model build successfully!')

    # Start training
    # Record training log
    log_epoch_idx = []
    recon_train = []
    sparse_train = []
    total_train = []
    recon_val = []
    sparse_val = []
    total_val = []
    training_log = {}

    print('***Start training!***')
    for epoch in range(args.epochs):
        lca.train()

        for batch_num, images in enumerate(tqdm(train_loader)):
            images = images.to(device=args.device)
            inputs, code, recon, recon_error = lca(images)
            lca.update_weights(code, recon_error)

        print(f'Training of epoch: {epoch + 1} finished!')

        if epoch == args.epochs - 1:

            print('Model performance:')

            # calculate loss on the training data
            L2_avg_recon_train, L1_avg_spar_train = evaluation(lca, train_loader, args, train_dataset.__len__())
            total_loss_train = L2_avg_recon_train + L1_avg_spar_train

            # calculate loss on the validation data
            L2_avg_recon_val, L1_avg_spar_val = evaluation(lca, vali_loader, args, vali_dataset.__len__())
            total_loss_val = L2_avg_recon_val + L1_avg_spar_val

            # Avg Recon Loss（平均重建损失）
            # Avg Sparsity Loss（平均稀疏损失）
            # Avg Total Loss（平均总损失）
            print(
                f'Training Set: Avg Recon Loss - {round(L2_avg_recon_train, 2)}; Avg Sparsity Loss - {round(L1_avg_spar_train, 2)}; Avg Total Loss - {round(total_loss_train, 2)}')
            print(
                f'Validation Set: Avg Recon Loss - {round(L2_avg_recon_val, 2)}; Avg Sparsity Loss - {round(L1_avg_spar_val, 2)}; Avg Total Loss - {round(total_loss_val, 2)}')

            log_epoch_idx.append(epoch + 1)
            recon_train.append(L2_avg_recon_train)
            sparse_train.append(L1_avg_spar_train)
            total_train.append(total_loss_train)
            recon_val.append(L2_avg_recon_val)
            sparse_val.append(L1_avg_spar_val)
            total_val.append(total_loss_val)

        # save parameters
        print('***Model parameters saved!***')
        torch.save(lca, f'{args.lca_model_saving_path}/SparseCoding_{args.ct}.pt')

    print('***Training finished!***')
    training_log['log_epoch_idx'] = log_epoch_idx
    training_log['recon_train'] = recon_train
    training_log['sparse_train'] = sparse_train
    training_log['total_train'] = total_train
    training_log['recon_val'] = recon_val
    training_log['sparse_val'] = sparse_val
    training_log['total_val'] = total_val

    # Start evaluation
    print('***Start testing!***')
    # load_best_model
    best_model = torch.load(f'{args.lca_model_saving_path}/SparseCoding_{args.ct}.pt')
    best_model.to(device=args.device)
    best_model.eval()

    L2_avg_recon_test, L1_avg_spar_test = evaluation(best_model, test_loader, args, test_dataset.__len__())
    total_loss_test = L2_avg_recon_test + L1_avg_spar_test

    training_log['recon_test'] = L2_avg_recon_test
    training_log['sparse_test'] = L1_avg_spar_test
    training_log['total_test'] = total_loss_test

    # Save experiment results
    file = open(f'{args.lca_model_saving_path}/training_log_{args.ct}.p', 'wb')
    pickle.dump(training_log, file)
    file.close()
    print('***All training, validation and testing logs are saved!***')
    # Finished!

# remove video with less than 100 frames
def remove_short_video(video_list, data_path, clip_frames):
    more_than_100 = []
    for V_id in video_list:
        # get path
        V_path = data_path + V_id + '/'
        # get frames num
        V_len = len(os.listdir(V_path))
        if V_len >= clip_frames:
            more_than_100.append(V_id)
    return more_than_100

def main():
    parser = argparse.ArgumentParser(description='Image based sparse coding model training')

    # data file settings
    parser.add_argument('--data_path', default=r'./datasets/')
    parser.add_argument('--data_labels', default=r'aligned_split.p')

    # Image preprocessing settings
    # parser.add_argument('--dim_h', default=320, type=int,
    #                     help='Dimension of input frame on h direction')
    # parser.add_argument('--dim_w', default=480, type=int,
    #                     help='Dimension of input frame on w direction')
    parser.add_argument('--dim_h', default=64, type=int,
                        help='Dimension of input frame on h direction')
    parser.add_argument('--dim_w', default=96, type=int,
                        help='Dimension of input frame on w direction')
    parser.add_argument('--in_chans', default=3, type=int,
                        help='Dimension of input channels, here is RGB=3')
    # reshape images into the shape of 320 x 480 x 3

    # LCA model settings
    parser.add_argument('--clip_frames', default=80, type=int)  # how many frames you want to include in a clip
    # parser.add_argument('--features', default=64, type=int,
    #                     help='Number of LCA neurons/features')  # 要多少特征图
    parser.add_argument('--features', default=32, type=int,
                        help='Number of LCA neurons/features')  # 要多少特征图
    parser.add_argument('--kernel_size', default=9, type=int,
                        help='Spatio-temporal size of the LCA receptive fields')
    parser.add_argument('--STRIDE', default=1, type=int,
                        help='Stride of the LCA receptive fields')
    parser.add_argument('--LAMBDA', default=0.25, type=float,
                        help='LCA firing threshold')
    parser.add_argument('--TAU', default=100, type=int,
                        help='LCA time constant')
    parser.add_argument('--track_metrics', default=False, type=bool,
                        help='Whether to track the L1 sparsity penalty, L2 reconstruction error, ' \
                             'total energy (L1 + L2), and fraction of neurons active over the LCA loop ' \
                             'at each forward pass and write it to a file in result_dir')
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help='Learning rate for built-in weight updates')
    parser.add_argument('--lca_model_saving_path', default='./Image_LCA')

    # model training settings
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    # parser.add_argument('--batch_size', default=48, type=int, help='batch size')
    # parser.add_argument('--epochs', default=30, type=int, help='number of total epochs to run')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--ct', default='xxx', type=str, help='current time')
    parser.add_argument('--SEED', default=2023, type=int, help='random state')

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device

    now = datetime.now()
    current_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    args.ct = current_time
    args.lca_model_saving_path = './Image_LCA_' + current_time
    # create folder
    folder = os.path.exists(args.lca_model_saving_path)
    if not folder:
        os.mkdir(args.lca_model_saving_path)

    # fix random state
    random.seed(args.SEED)
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed(args.SEED)

    # load dataset pickles file
    pickle_file_path = args.data_labels
    dataset_sep = pickle.load(open(pickle_file_path, "rb"))
    # train set
    train_list = dataset_sep['train'].keys()  # 001102, 000920, 001268, ...
    # val set
    val_list = dataset_sep['val'].keys()  # 001102, 000920, 001268, ...
    # test set
    test_list = dataset_sep['test'].keys()  # 001102, 000920, 001268, ...

    # clean dataset
    args.train_list = remove_short_video(train_list, args.data_path, args.clip_frames)
    args.val_list = remove_short_video(val_list, args.data_path, args.clip_frames)
    args.test_list = remove_short_video(test_list, args.data_path, args.clip_frames)

    train(args)


if __name__=='__main__':
    main()
