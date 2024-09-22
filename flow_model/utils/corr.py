import torch
import torch.nn.functional as F
from utils.utils import bilinear_sampler, coords_grid

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation 生成代价空间
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)
        # 将上面的corr添加到代价金字塔
        self.corr_pyramid.append(corr)
        # 再添加三次下采样后的corr
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        # 将坐标进行转置操作，以使其与相关性金字塔的形状匹配
        coords = coords.permute(0, 2, 3, 1)
        # 获取相关性金字塔的每个层次的尺寸
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            # 计算用于坐标偏移的delta向量
            dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)

            # 将坐标展平为一维向量
            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            # 将偏移向量重塑为（1, 2r+1, 2r+1, 2）的形状，以便它可以与每个坐标向量进行广播
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            # 将偏移向量添加到坐标向量上，以获得偏移后的坐标
            coords_lvl = centroid_lvl + delta_lvl

            # 使用bilinear_sampler()函数根据偏移后的坐标从相关性金字塔的每个层次上进行插值
            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)
            
        # 将相关性金字塔的每个层次的结果连接在一起
        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod   # 设置为静态方法，通过类名调用，而无需实例化类（corr）。这使得代码更简洁，易于维护
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd) 
        # 将fmap1转置后与fmap2相乘得到代价空间
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        # 对相关性进行归一化 并返回结果
        return corr  / torch.sqrt(torch.tensor(dim).float())


class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())
