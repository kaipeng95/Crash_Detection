import argparse
import torch
from raft import RAFT

def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to('cuda')
    model.eval()
    print(model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='checkpoints/raft-kitti.pth', help="restore checkpoint")
    # parser.add_argument('--path', default='demo-frames', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    # parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    # parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)