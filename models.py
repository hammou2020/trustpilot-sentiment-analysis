import torch
from torch import nn
import torch.nn.functional as F


class SentimentClassifier(nn.Module):
    def __init__(self, max_tokens,
                 conv_num_kernels=[256] * 6,
                 conv_kernel_sizes=[7, 7, 3, 3, 3, 3],
                 pool_sizes=[3, 3, None, None, None, 3],
                 fc_sizes=[1024, 1024, 3]):
        super(SentimentClassifier, self).__init__()
        self.conv1ds = nn.ModuleList()
        self.pools = nn.ModuleList([None] * len(conv_num_kernels))
        self.fcs = nn.ModuleList()

        in_c = max_tokens
        for i in range(len(conv_num_kernels)):
            out_c = conv_num_kernels[i]
            kernel_size = conv_kernel_sizes[i]
            self.conv1ds.append(
                nn.Conv1d(in_c, out_c, kernel_size))
            in_c = out_c

            if pool_sizes[i]:
                self.pools[i] = nn.MaxPool1d(pool_sizes[i])
        in_feats = 768
        for fc_size in fc_sizes:
            self.fcs.append(nn.Linear(in_feats, fc_size))
            in_feats = fc_size

    def forward(self, x):
        batch_size = x.shape[0]
        for i, conv in enumerate(self.conv1ds):
            x = conv(x)
            if self.pools[i]:
                x = self.pools[i](x)
            # print(x.shape)
        x = x.reshape(batch_size, -1)

        for fc in self.fcs:
            x = fc(x)
#         x = F.softmax(x, dim=-1)

        return x
