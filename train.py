import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from shutil import copyfile
from torch.utils import data

from Criterions import CombinedLoss
from FaceTripletsLoader import FaceTripletsSet, FaceTriplets
from LFWpairsLoader import LFWDataset, LFWMaskedDataset
from Models.inception_resnet_v1 import InceptionResnetV1
from evaluate_LFW import evaluate_lfw

import matplotlib.pyplot as plt
import seaborn as sb

SAVE_FOLDER = './Saved_CombineLosser/IRN1_tanhnorm_SGD-StepLR_freezed'
CHECKPOINT_FILE = os.path.join(SAVE_FOLDER, 'model-checkpoint.tar')

TRAIN_FOLDER = './Data/CASIA-masked'
LFW_FOLDER = './Data/LFW/lfw-masked'
LFW_PAIRSFILE = './Data/LFW/LFW_pairs.txt'

EPOCHS = 20
BATCH_SIZE = 32
MARGIN = 0.5


def train(model, dataloader, criterion, optimizer, scheduler, distancer):
    model.train()
    avg_loss = 0
    num_hard_triplets = 0
    with tqdm(dataloader, desc='Training') as iterator:
        for i, batch in enumerate(iterator):
            model.zero_grad()

            anc_img = batch['anc_img']
            pos_img = batch['pos_img']
            neg_img = batch['neg_img']

            pos_cls = batch['pos_cls']
            neg_cls = batch['neg_cls']

            (anc_emb, anc_lgt), (pos_emb, pos_lgt), (neg_emb, neg_lgt) = \
                model(anc_img.cuda()), model(pos_img.cuda()), model(neg_img.cuda())

            pos_dis = distancer(anc_emb, pos_emb)
            neg_dis = distancer(anc_emb, neg_emb)
            dis = (neg_dis - pos_dis < MARGIN).detach().cpu().numpy().flatten()

            hard_triplets = np.where(dis == 1)[0]
            if len(hard_triplets) == 0:  # Train only when have a hard triplets.
                continue

            anc_emb = anc_emb[hard_triplets]
            pos_emb = pos_emb[hard_triplets]
            neg_emb = neg_emb[hard_triplets]

            loss = criterion(anc_emb, pos_emb, neg_emb,
                             torch.cat([anc_lgt, pos_lgt, neg_lgt]),
                             torch.cat([pos_cls, pos_cls, neg_cls]).cuda().squeeze())

            avg_loss += loss.item()
            num_hard_triplets += len(hard_triplets)

            loss.backward()
            optimizer.step()
            scheduler.step()

            iterator.set_postfix_str(' Loss: {:.4f}'.format(loss.item()))
            iterator.update()

    avg_loss = 0 if num_hard_triplets == 0 else avg_loss / num_hard_triplets
    return avg_loss, num_hard_triplets


def get_lfw_distances(model, dataloader, distancer):
    model.eval()
    distances, labels = [], []
    with tqdm(dataloader, desc='Calculate Distances') as iterator:
        for img_a, img_b, label in iterator:
            (out_a, lgt_a), (out_b, lgt_b) = model(img_a.cuda()), model(img_b.cuda())
            distance = distancer(out_a, out_b)

            distances.extend(distance.detach().cpu().numpy())
            labels.extend(label.numpy())
            iterator.update()

    distances = np.array(distances)
    labels = np.array(labels)
    return distances, labels


def evaluate(model, dataloader, distancer, title='', save=None):
    distances, labels = get_lfw_distances(model, dataloader, distancer)

    metrics = evaluate_lfw(distances=distances, labels=labels)
    txt = "Accuracy on LFW: {:.4f}+-{:.4f}\nPrecision {:.4f}+-{:.4f}\nRecall {:.4f}+-{:.4f}" \
          "\nROC Area Under Curve: {:.4f}\nBest distance threshold: {:.2f}+-{:.2f}" \
          "\nTAR: {:.4f}+-{:.4f} @ FAR: {:.4f}".format(
            np.mean(metrics['accuracy']),
            np.std(metrics['accuracy']),
            np.mean(metrics['precision']),
            np.std(metrics['precision']),
            np.mean(metrics['recall']),
            np.std(metrics['recall']),
            metrics['roc_auc'],
            np.mean(metrics['best_distances']),
            np.std(metrics['best_distances']),
            np.mean(metrics['tar']),
            np.std(metrics['tar']),
            np.mean(metrics['far']))
    print(txt)

    fig, axes = plt.subplots(1, 2)
    fig.suptitle(title, fontsize=15)
    fig.set_size_inches(14, 6)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    axes[0].set_title('distance histogram')
    sb.distplot(distances[labels == True], ax=axes[0], label='distance-true')
    sb.distplot(distances[labels == False], ax=axes[0], label='distance-false')
    axes[0].legend()

    axes[1].text(0.05, 0.3, txt, fontsize=20)
    axes[1].set_axis_off()
    if save is not None:
        plt.savefig(save)
        plt.close()

    return metrics


if __name__ == '__main__':
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    copyfile('train.py', os.path.join(SAVE_FOLDER, 'train.py'))

    transforms_fn = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(
                                            mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5]
                                        )])
    train_loader = data.DataLoader(
        dataset=FaceTriplets(TRAIN_FOLDER, transform=transforms_fn),
        batch_size=BATCH_SIZE,
        num_workers=6,
        shuffle=True
    )
    lfw_loader = data.DataLoader(
        dataset=LFWDataset(LFW_FOLDER, LFW_PAIRSFILE, transform=transforms_fn),
        batch_size=32,
        num_workers=4,
        shuffle=False
    )
    lfw_mask_loader = data.DataLoader(
        dataset=LFWDataset(LFW_FOLDER, LFW_PAIRSFILE, mask=True, transform=transforms_fn),
        batch_size=32,
        num_workers=4,
        shuffle=True
    )

    model = InceptionResnetV1('casia-webface', classify=True, device='cuda')
    # Freeze some layers.
    for layer in list(model.children())[:-5]:
        for p in layer.parameters():
            p.requires_grad = False

    criterion = CombinedLoss(MARGIN)

    #optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-6)
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    l2_distance = nn.PairwiseDistance(2.)

    loss_list = []
    num_train_list = []
    metric_list = []
    start_epoch = 0
    best_accuracy = 0

    if os.path.exists(CHECKPOINT_FILE):
        print('Continue training.')
        state = torch.load(CHECKPOINT_FILE)
        loss_list = state['loss_list']
        num_train_list = state['num_train_list']
        metric_list = state['metric_list']
        start_epoch = state['epoch']
        best_accuracy = state['best_accuracy']
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optim_state_dict'])
        lr_scheduler.load_state_dict(state['lr_scheduler_state_dict'])

    for e in range(start_epoch, EPOCHS):
        print('Epoch {}/{}'.format(e, EPOCHS - 1))
        ep_st = time.time()

        ep_loss, ep_num_train = train(model, train_loader, criterion, optimizer, lr_scheduler, l2_distance)
        print('Epoch train loss avg: {:.4f}, Num train: {}'.format(ep_loss, ep_num_train))
        loss_list.append(ep_loss)
        num_train_list.append(ep_num_train)

        ep_metric = evaluate(model, lfw_mask_loader, l2_distance,
                             title='LFW with mask ep: {}'.format(e),
                             save=os.path.join(SAVE_FOLDER, 'lfw-withmask_ep-{}.png'.format(e)))
        metric_list.append(ep_metric)

        if ep_metric['accuracy'].mean() > best_accuracy:
            best_accuracy = ep_metric['accuracy'].mean()
            state = {'epoch': e + 1,
                     'loss_list': loss_list,
                     'num_train_list': num_train_list,
                     'metric_list': metric_list,
                     'best_accuracy': best_accuracy,
                     'model_state_dict': model.state_dict(),
                     'optim_state_dict': optimizer.state_dict(),
                     'lr_scheduler_state_dict': lr_scheduler.state_dict()}
            torch.save(state, os.path.join(SAVE_FOLDER, 'model-checkpoint.tar'))

        ep_et = time.time() - ep_st
        print('Epoch total-time used: %.0f h : %.0f m : %.0f s' %
              (ep_et // 3600, ep_et // 60, ep_et % 60))
