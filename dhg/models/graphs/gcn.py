import torch
import torch.nn as nn

import dhg
from dhg.nn import GCNConv


class GCN(nn.Module):
    r"""The GCN model proposed in `Semi-Supervised Classification with Graph Convolutional Networks <https://arxiv.org/pdf/1609.02907>`_ paper (ICLR 2017).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): Dropout ratio. Defaults to ``0.5``.
    """
    def __init__(self, in_channels: int,
                 hid_channels: int,
                 num_classes: int,
                 use_bn: bool = False,
                 drop_rate: float = 0.5) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate))
        self.layers.append(GCNConv(hid_channels, num_classes, use_bn=use_bn, is_last=True))


    def train_layer_0(self, X: torch.Tensor, g: "dhg.Graph") -> torch.Tensor:
        out = self.layers[0](X, g)
        return out

    def train_layer_1(self, X: torch.Tensor, g: "dhg.Graph") -> torch.Tensor:
        out = self.layers[1](X, g)
        return out

    def forward(self, X: torch.Tensor, g: "dhg.Graph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``g`` (``dhg.Graph``): The graph structure that contains :math:`N` vertices.
        """
        outs = []
        for layer in self.layers:
            X = layer(X, g)
            outs.append(X)
        return outs


class TGCN(nn.Module):
    r"""The GCN model proposed in `Semi-Supervised Classification with Graph Convolutional Networks <https://arxiv.org/pdf/1609.02907>`_ paper (ICLR 2017).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): Dropout ratio. Defaults to ``0.5``.
    """
    def __init__(self, in_channels: int,
                 hid_channels: int,
                 num_classes: int,
                 temperature: float = 0.5,
                 use_bn: bool = False,
                 drop_rate: float = 0.5,
                 ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate))
        # self.layers.append(GCNConv(hid_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate))
        # self.layers.append(GCNConv(hid_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate))
        self.layers.append(GCNConv(hid_channels, num_classes, use_bn=use_bn, is_last=True))
        self.temperature = temperature
        self.x_pre = 0


    def forward(self, X: torch.Tensor, g: "dhg.Graph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``g`` (``dhg.Graph``): The graph structure that contains :math:`N` vertices.
        """
        outs = []
        layerCount = 0
        for layer in self.layers:
            X = layer(X, g)
            layerCount += 1
            # if layerCount == 1:
            #     X = X + self.feaIntervene   # do intervene for feature
            outs.append(X)
        return outs

    def getMoonCrossLoss(self, epoch, x_l, x_g):
        if epoch == 0:
            self.x_pre = x_l.detach()
        posSim = torch.cosine_similarity(x_l, x_g, dim=1)
        posSim = torch.exp((posSim / self.temperature)/100)
        posSim = posSim.sum(dim=0, keepdim=True)

        negSim = torch.cosine_similarity(x_l, self.x_pre, dim=1)
        negSim = negSim.sum(dim=0, keepdim=True)
        negSim = posSim + negSim

        self.x_pre = x_l.detach()
        loss = (-torch.log(posSim / negSim)).mean()
        return loss


    def getCrossLoss(self, x_l, x_g, label):  # local and global, 同一类中的节点为正例，不同类中的节点为反例
        curItem = x_l[0]
        curLbl = label[0]
        posIndex = [i for i, lbl in enumerate(label) if lbl == curLbl]
        negIndex = [i for i, lbl in enumerate(label) if lbl != curLbl]
        posItems = x_g[posIndex]
        negItems = x_g[negIndex]

        curItemList = curItem.repeat([len(posIndex), 1])
        posSim = torch.cosine_similarity(curItemList, posItems, dim=1)
        posSim = torch.exp(((posSim / self.temperature)/100))
        posSim = posSim.sum(dim=0, keepdim=True)

        curItemList = curItem.repeat([len(negIndex), 1])
        negSim = torch.cosine_similarity(curItemList, negItems, dim=1)
        negSim = torch.exp(((negSim / self.temperature)/100))
        negSim = negSim.sum(dim=0, keepdim=True)

        for i in range(1, len(x_l)):
            curItem = x_l[i]
            curLbl = label[i]
            posIndex = [i for i, lbl in enumerate(label) if lbl == curLbl]
            negIndex = [i for i, lbl in enumerate(label) if lbl != curLbl]
            posItems = x_g[posIndex]
            negItems = x_g[negIndex]

            curItemList = curItem.repeat([len(posIndex), 1])
            eachSim = torch.cosine_similarity(curItemList, posItems, dim=1)
            eachSim = torch.exp((eachSim / self.temperature)/100)
            eachSim = eachSim.sum(dim=0, keepdim=True)
            posSim = torch.cat((posSim, eachSim), dim=0)

            curItemList = curItem.repeat([len(negIndex), 1])
            eachSim = torch.cosine_similarity(curItemList, negItems, dim=1)
            eachSim = torch.exp((eachSim / self.temperature)/100)
            eachSim = eachSim.sum(dim=0, keepdim=True)
            negSim = torch.cat((negSim, eachSim), dim=0)

        loss = (-torch.log(posSim / negSim)).mean()
        return loss


