import torch
import torch.nn as nn


class CombinedLoss(nn.Module):
    """Combination of Triplet Loss and CrossEntropy Loss."""
    def __init__(self, margin=0.5):
        super(CombinedLoss, self).__init__()
        self.tpl = nn.TripletMarginLoss(margin=margin)
        self.cnl = nn.CrossEntropyLoss()

    def forward(self, anc_emb, pos_emb, neg_emb, pred_cls, true_cls):
        loss_tpl = self.tpl(anc_emb, pos_emb, neg_emb)
        loss_cnl = self.cnl(pred_cls, true_cls)
        return loss_tpl + loss_cnl
