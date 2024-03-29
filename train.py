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

from augmentation import face_augment_pipe
from FaceMasking import AugmentMasking
from FaceTripletsLoader import FaceTriplets
from LFWpairsLoader import LFWDataset
from Models.inception_resnet_v1 import InceptionResnetV1
from Criterions import CombinedLoss
from evaluate_LFW import evaluate_lfw

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sb

SAVE_FOLDER = './Experiments2/IRN1-freezed-5_combine_Adam1e4_mc025'
CHECKPOINT_FILE = os.path.join(SAVE_FOLDER, 'model-checkpoint.tar')

TRAIN_FOLDER = ['../Data/CASIA-WebFace', '../Data/RMFD/AFDB_face_dataset']
VALID_FOLDER = '../Data/LFW/lfw-masked'
PAIRS_FILE = '../Data/LFW/LFW_pairs.txt'

PRETRAINED = 'vggface2'
IMAGE_SIZE = 160

EPOCHS = 20
BATCH_SIZE = 32
MARGIN = 0.5
MASK_CHANCE = 0.25
MASK_PTS_FILE = './mask_images/mask_pts.pkl'


def main():
    torch.backends.cudnn.benchmark = True

    transforms_fn = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(
                                            mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
                                        ])

    train_loader = data.DataLoader(
        dataset=FaceTriplets(TRAIN_FOLDER, image_size=IMAGE_SIZE,
                             augment=AugmentMasking(mask_chance=MASK_CHANCE,
                                                    post_augment=None,
                                                    mask_pts_file=MASK_PTS_FILE),
                             transform=transforms_fn),
        batch_size=BATCH_SIZE,
        num_workers=5,
        shuffle=True
    )
    num_classes = len(train_loader.dataset.data['class'].unique())

    valid_mask_loader = data.DataLoader(
        dataset=LFWDataset(VALID_FOLDER, PAIRS_FILE, image_size=IMAGE_SIZE,
                           mask=True, transform=transforms_fn),
        batch_size=BATCH_SIZE,
        num_workers=2,
        shuffle=False
    )
    valid_nomask_loader = data.DataLoader(
        dataset=LFWDataset(VALID_FOLDER, PAIRS_FILE, image_size=IMAGE_SIZE,
                           mask=False, transform=transforms_fn),
        batch_size=BATCH_SIZE,
        num_workers=2,
        shuffle=False
    )

    model = InceptionResnetV1(PRETRAINED, classify=True, num_classes=num_classes, device='cuda')
    # Freeze some layers.
    for layer in list(model.children())[:-5]:
        for p in layer.parameters():
            p.requires_grad = False

    #optimizer = optim.Adadelta(model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-6)
    #optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    #lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3)
    l2_distance = nn.PairwiseDistance(2.)

    criterion = CombinedLoss(MARGIN)

    loss_names = criterion._get_name()
    loss_names = loss_names + ['total'] if isinstance(loss_names, list) else [loss_names]

    loss_list = {ln: [] for ln in loss_names}
    num_hard_triplets_list = []
    num_has_mask = []
    metric_list = {'mask': [], 'nomask': []}
    start_epoch = 0
    best_tar = 0
    best_state = None

    if os.path.exists(CHECKPOINT_FILE):
        print('Continue training.')
        state = torch.load(CHECKPOINT_FILE)
        loss_list = state['loss_list']
        num_hard_triplets_list = state['num_hard_triplets_list']
        num_has_mask = state['num_has_mask']
        metric_list = state['metric_list']
        start_epoch = state['epoch']
        best_tar = state['best_tar']
        best_state = state['best_state']
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optim_state_dict'])
        #lr_scheduler.load_state_dict(state['lr_scheduler_state_dict'])

    for e in range(start_epoch, EPOCHS):
        print('Epoch {}/{}'.format(e, EPOCHS - 1))
        ep_st = time.time()

        # Train.
        ep_loss, ep_num_hard_triplets, ep_num_has_mask = train(
            model, train_loader, criterion, optimizer, l2_distance, loss_names)
        txt = 'Train Avg Loss | '
        for k, v in ep_loss.items():
            loss_list[k].append(v)
            txt += '{}: {:.4f},'.format(k, v)
        num_hard_triplets_list.append(ep_num_hard_triplets)
        num_has_mask.append(ep_num_has_mask)
        print(txt)
        print('Num of Hard-Triplets: {}'.format(ep_num_hard_triplets))
        print('Num of has Masked: {}'.format(ep_num_has_mask))
        print()

        # Validate.
        print('Evaluate with No-Mask.')
        ep_metric = evaluate(model, valid_nomask_loader, l2_distance,
                             title='LFW no mask ep: {}'.format(e),
                             save=os.path.join(SAVE_FOLDER, 'lfw-nomask_ep-{}.png'.format(e)))
        metric_list['nomask'].append(ep_metric)

        print('Evaluate with Mask.')
        ep_metric = evaluate(model, valid_mask_loader, l2_distance,
                             title='LFW with mask ep: {}'.format(e),
                             save=os.path.join(SAVE_FOLDER, 'lfw-withmask_ep-{}.png'.format(e)))
        metric_list['mask'].append(ep_metric)

        if ep_metric['tar'].mean() > best_tar:
            best_tar = ep_metric['tar'].mean()
            best_state = model.state_dict()

        # Save.
        state = {'epoch': e + 1,
                 'loss_list': loss_list,
                 'num_hard_triplets_list': num_hard_triplets_list,
                 'num_has_mask': num_has_mask,
                 'metric_list': metric_list,
                 'best_tar': best_tar,
                 'best_state': best_state,
                 'model_state_dict': model.state_dict(),
                 'optim_state_dict': optimizer.state_dict(),
                 #'lr_scheduler_state_dict': lr_scheduler.state_dict()
                 }
        torch.save(state, os.path.join(SAVE_FOLDER, 'model-checkpoint.tar'))

        #lr_scheduler.step()
        #print('lr reduce to : {}'.format(optimizer.param_groups[0]['lr']))

        ep_et = time.time() - ep_st
        print('Epoch total-time used: %.0f h : %.0f m : %.0f s' %
              (ep_et // 3600, ep_et // 60, ep_et % 60))


def train(model, dataloader, criterion, optimizer, distancer, loss_names):
    model.train()
    avg_loss = {ln: 0 for ln in loss_names}
    num_hard_triplets = 0
    num_has_mask = 0
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

            # use same samples as triplet for cross-entropy.
            anc_lgt = anc_lgt[hard_triplets]
            pos_lgt = pos_lgt[hard_triplets]
            neg_lgt = neg_lgt[hard_triplets]
            pos_cls = pos_cls[hard_triplets]
            neg_cls = neg_cls[hard_triplets]

            losses = criterion(anc_emb, pos_emb, neg_emb,
                               torch.cat([anc_lgt, pos_lgt, neg_lgt]),
                               torch.cat([pos_cls, pos_cls, neg_cls]).cuda().squeeze())
            txt = ''
            if isinstance(losses, tuple):
                loss = 0
                for j, ls in enumerate(losses):
                    loss += ls
                    avg_loss[loss_names[j]] += ls.item()
                    txt += '{}: {:.4f}, '.format(loss_names[j], ls.item())
                txt += 'Total: {:.4f}'.format(loss.item())
                loss.backward()
            else:
                avg_loss[loss_names[0]] += losses.item()
                txt = '{}: {:.4f}'.format(loss_names[0], losses.item())
                losses.backward()

            optimizer.step()

            num_hard_triplets += len(hard_triplets)
            num_has_mask += batch['is_mask'][hard_triplets].sum()

            iterator.set_postfix_str(txt)
            iterator.update()

    for ln in avg_loss.keys():
        avg_loss[ln] = 0 if num_hard_triplets == 0 else avg_loss[ln] / num_hard_triplets
    if 'total' in loss_names:
        avg_loss['total'] = sum([v for v in avg_loss.values()])
    return avg_loss, num_hard_triplets, num_has_mask


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
    print()

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
    copyfile('augmentation.py', os.path.join(SAVE_FOLDER, 'augmentation.py'))
    copyfile('FaceMasking.py', os.path.join(SAVE_FOLDER, 'FaceMasking.py'))
    copyfile('FaceTripletsLoader.py', os.path.join(SAVE_FOLDER, 'FaceTripletsLoader.py'))

    main()
