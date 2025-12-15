# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import argparse
import time
import shutil
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import os.path as osp
import csv
import numpy as np

np.random.seed(1337)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from model_ImpSGNv2 import ImpSGNv2
from data import NTUDataLoaders, AverageMeter
import fit
from util import make_dir, get_num_classes
######################### to plot the Confusion Matrix
############# for the confusion matrix:  (add these imports at the top)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
##############

parser = argparse.ArgumentParser(description='Skeleton-Based Action Recgnition')
fit.add_fit_args(parser)
parser.set_defaults(
    network ='ImpSGNv2', 
    dataset = 'NTU', #'NTU120',
    case = 0,
    batch_size=64,
    max_epochs=140,
    monitor='val_acc',
    lr=0.001,
    weight_decay=0.0001,
    lr_factor=0.1,
    workers=16,
    print_freq = 20,
    train = 0,
    seg = 20,
    )
args = parser.parse_args()




def main():

    args.num_classes = get_num_classes(args.dataset)
    model = ImpSGNv2(args.num_classes, args.dataset, args.seg, args)

    total = get_n_params(model)
    print(model)
    print('The number of parameters: ', total)
    print('The modes is:', args.network)

    if torch.cuda.is_available():
        print('It is using GPU!')
        model = model.cuda()

    criterion = LabelSmoothingLoss(args.num_classes, smoothing=0.1).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.monitor == 'val_acc':
        mode = 'max'
        monitor_op = np.greater
        best = -np.Inf
        str_op = 'improve'
    elif args.monitor == 'val_loss':
        mode = 'min'
        monitor_op = np.less
        best = np.Inf
        str_op = 'reduce'

    scheduler = MultiStepLR(optimizer, milestones=[60, 100, 120], gamma=0.1)
    # Data loading
    ntu_loaders = NTUDataLoaders(args.dataset, args.case, seg=args.seg)
    train_loader = ntu_loaders.get_train_loader(args.batch_size, args.workers)
    val_loader = ntu_loaders.get_val_loader(args.batch_size, args.workers)
    train_size = ntu_loaders.get_train_size()
    val_size = ntu_loaders.get_val_size()


    test_loader = ntu_loaders.get_test_loader(32, args.workers)	#if you change '32' => change '32 * 5' in model.py 

    print('Train on %d samples, validate on %d samples' % (train_size, val_size))

    best_epoch = 0
    output_dir = make_dir(args.dataset)

    save_path = os.path.join(output_dir, args.network)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    checkpoint = osp.join(save_path, '%s_best.pth' % args.case)
    earlystop_cnt = 0
    csv_file = osp.join(save_path, '%s_log.csv' % args.case)
    log_res = list()

    lable_path = osp.join(save_path, '%s_lable.txt'% args.case)
    pred_path = osp.join(save_path, '%s_pred.txt' % args.case)


    # Training
    if args.train ==1:
        for epoch in range(args.start_epoch, args.max_epochs):

            print(epoch, optimizer.param_groups[0]['lr'])

            t_start = time.time()
            train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch)
            val_loss, val_acc = validate(val_loader, model, criterion)
            log_res += [[train_loss, train_acc.cpu().numpy(),\
                         val_loss, val_acc.cpu().numpy()]]

            print('Epoch-{:<3d} {:.1f}s\t'
                  'Train: loss {:.4f}\taccu {:.4f}\tValid: loss {:.4f}\taccu {:.4f}'
                  .format(epoch + 1, time.time() - t_start, train_loss, train_acc, val_loss, val_acc))

            current = val_loss if mode == 'min' else val_acc

            ####### store tensor in cpu
            current = current.cpu()

            if monitor_op(current, best):
                print('Epoch %d: %s %sd from %.4f to %.4f, '
                      'saving model to %s'
                      % (epoch + 1, args.monitor, str_op, best, current, checkpoint))
                best = current
                best_epoch = epoch + 1
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best': best,
                    'monitor': args.monitor,
                    'optimizer': optimizer.state_dict(),
                }, checkpoint)
                earlystop_cnt = 0
            else:
                print('Epoch %d: %s did not %s' % (epoch + 1, args.monitor, str_op))
                earlystop_cnt += 1

            scheduler.step()

        print('Best %s: %.4f from epoch-%d' % (args.monitor, best, best_epoch))
        with open(csv_file, 'w') as fw:
            cw = csv.writer(fw)
            cw.writerow(['loss', 'acc', 'val_loss', 'val_acc'])
            cw.writerows(log_res)
        print('Save train and validation log into into %s' % csv_file)

    ### Test
    args.train = 0
    model = ImpSGNv2(args.num_classes, args.dataset, args.seg, args)
    model = model.cuda()
#    test(test_loader, model, checkpoint, lable_path, pred_path)
    ################### adding the Confusion Matrix calculus after the test()
    # Plot and save the confusion matrix
    class_names = [str(i) for i in range(args.num_classes)]  # Replace with actual names if available
    cm_save_path = os.path.join(save_path, 'confusion_matrix.png')
#    plot_confusion_matrix(lable_path, pred_path, save_path=cm_save_path, class_names=ntu60_actions)
    plot_confusion_matrix(lable_path, pred_path, save_path=cm_save_path)


def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    acces = AverageMeter()
    model.train()

    for i, (inputs, target) in enumerate(train_loader):

        output = model(inputs.cuda())
        target = target.cuda()
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc = accuracy(output.data, target)
        losses.update(loss.item(), inputs.size(0))
        acces.update(acc[0], inputs.size(0))

        # backward
        optimizer.zero_grad()  # clear gradients out before each mini-batch
        loss.backward()
        optimizer.step()

        if (i + 1) % args.print_freq == 0:
            print('Epoch-{:<3d} {:3d} batches\t'
                  'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'accu {acc.val:.3f} ({acc.avg:.3f})'.format(
                   epoch + 1, i + 1, loss=losses, acc=acces))

    return losses.avg, acces.avg


def validate(val_loader, model, criterion):
    losses = AverageMeter()
    acces = AverageMeter()
    model.eval()

    for i, (inputs, target) in enumerate(val_loader):
        with torch.no_grad():
            output = model(inputs.cuda())
        target = target.cuda()
        with torch.no_grad():
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc = accuracy(output.data, target)
        losses.update(loss.item(), inputs.size(0))
        acces.update(acc[0], inputs.size(0))

    return losses.avg, acces.avg


def test(test_loader, model, checkpoint, lable_path, pred_path):
    acces = AverageMeter()
    # load learnt model that obtained best performance on validation set
    model.load_state_dict(torch.load(checkpoint)['state_dict'])
    model.eval()

    label_output = list()
    pred_output = list()

    t_start = time.time()
    for i, (inputs, target) in enumerate(test_loader):
        with torch.no_grad():
            output = model(inputs.cuda())
            output = output.view((-1, inputs.size(0)//target.size(0), output.size(1)))
            output = output.mean(1)

        label_output.append(target.cpu().numpy())
        pred_output.append(output.cpu().numpy())

        acc = accuracy(output.data, target.cuda())
        acces.update(acc[0], inputs.size(0))


    label_output = np.concatenate(label_output, axis=0)
    np.savetxt(lable_path, label_output, fmt='%d')
    pred_output = np.concatenate(pred_output, axis=0)
    np.savetxt(pred_path, pred_output, fmt='%f')

    print('Test: accuracy {:.3f}, time: {:.2f}s'
          .format(acces.avg, time.time() - t_start))
###################################################################### adding the confusion matrix:

### Version 6 with action names (works) :
def plot_confusion_matrix(label_path, pred_path, save_path='confusion_matrix.png'):
    #### Uncomment this block if using NTU  60
    class_names = [
    'drink', 'eat', 'brushing_teeth', 'brushing_hair', 'drop', 'pickup',
    'throw', 'sitting_down', 'standing_up', 'clapping', 'reading', 'writing', 'tear_up_paper',
    'wear_jacket', 'take_off_jacket', 'wear_shoe', 'take_off_shoe', 'wear_on_glasses',
    'take_off_glasses', 'put_on_hat', 'take_off_hat', 'cheer_up', 'hand_waving',
    'kicking_something', 'reach_into_pocket', 'hopping', 'jump_up', 'make_phone_call',
    'play_with_phone', 'type_on_a_keyboard', 'point_to_something', 'taking a selfie',
    'check time (watch)', 'rub two hands together', 'nod head/bow', 'shake head',
    'wipe face', 'salute', 'put the palms together', 'cross hands in front',
    'sneeze/cough', 'staggering', 'falling', 'touch head', 'touch chest', 'touch back',
    'touch neck', 'nausea or vomiting', 'use a fan (with hand or object)', 'punching/slapping other person',
    'kicking other person', 'pushing other person', 'pat on back of other person',
    'point finger at the other person', 'hugging other person', 'giving something to other person',
    'touch other person\'s pocket', 'handshaking', 'walking towards each other',
    'walking apart from each other'
    ]
    

    """ #### Uncomment this block if using NTU120
    class_names = [
    # A1–A60 (same as NTU‑60):
    'drink_water','eat_meal/snack','brushing_teeth','brushing_hair','drop','pickup',
    'throw','sitting_down','standing_up','clapping','reading','writing','tear_up_paper',
    'wear_jacket','take_off_jacket','wear_shoe','take_off_shoe','wear_on_glasses',
    'take_off_glasses','put_on_hat/cap','take_off_hat/cap','cheer_up','hand_waving',
    'kicking_something','reach_into_pocket','hopping','jump_up','make_phone_call',
    'play_with_phone','type_on_keyboard','point_to_something','taking_selfie',
    'check_time_watch','rub_hands_together','nod_head_bow','shake_head',
    'wipe_face','salute','put_palms_together','cross_hands_in_front','sneeze_cough',
    'staggering','falling','touch_head','touch_chest','touch_back','touch_neck',
    'nausea_vomiting','use_fan_hand_or_object','punching_slapping_other_person',
    'kicking_other_person','pushing_other_person','pat_on_back_other_person',
    'point_finger_other_person','hugging_other_person','giving_something_to_other_person',
    'touch_other_persons_pocket','handshaking','walking_towards_each_other',
    'walking_apart_from_each_other',
    # A61–A120 (new NTU‑120 classes) :contentReference[oaicite:1]{index=1}:
    'put_on_headphone','take_off_headphone','shoot_at_basket','bounce_ball',
    'tennis_bat_swing','juggling_table_tennis_balls','hush_quiet','flick_hair','thumb_up',
    'thumb_down','make_ok_sign','make_victory_sign','staple_book','counting_money',
    'cutting_nails','cutting_paper_using_scissors','snapping_fingers','open_bottle',
    'sniff_smell','squat_down','toss_a_coin','fold_paper','ball_up_paper',
    'play_magic_cube','apply_cream_on_face','apply_cream_on_hand_back','put_on_bag',
    'take_off_bag','put_something_into_bag','take_something_out_of_bag','open_a_box',
    'move_heavy_objects','shake_fist','throw_up_cap/hat','hands_up_both_hands',
    'cross_arms','arm_circles','arm_swings','running_on_the_spot','butt_kicks_kick_backward',
    'cross_toe_touch','side_kick','yawn','stretch_oneself','blow_nose',
    'hit_other_person_with_something','wield_knife_towards_other_person',
    'knock_over_other_person','grab_other_persons_stuff','shoot_at_other_person_with_a_gun',
    'step_on_foot','high-five','cheers_and_drink','carry_something_with_other_person',
    'take_a_photo_of_other_person','follow_other_person','whisper_in_other_persons_ear',
    'exchange_things_with_other_person','support_somebody_with_hand','finger-guessing_game'
]"""

    labels = np.loadtxt(label_path, dtype=int)
    preds = np.loadtxt(pred_path)
    preds = np.argmax(preds, axis=1)

    # Adjust for 1-based indexing if necessary
    if labels.min() == 1 and labels.max() == 60:
        labels -= 1

    assert np.max(labels) < len(class_names), "Label index exceeds number of class names"
    assert np.min(labels) >= 0, "Negative label detected"

    # Compute and normalize confusion matrix
    cm = confusion_matrix(labels, preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(36, 36))
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Axis labels and ticks
    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           title='Normalized Confusion Matrix - NTU RGB+D 60',
           ylabel='True Label',
           xlabel='Predicted Label')

    plt.setp(ax.get_xticklabels(), rotation=90) #, ha="center", rotation_mode="anchor"
    plt.setp(ax.get_yticklabels(), rotation=0)

    # Format cell annotations
    fmt = ".2f"
    thresh = cm_normalized.max() / 2.
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            value = format(cm_normalized[i, j], fmt)
            ax.text(j, i, value, ha="center", va="center",
                    color="white" if cm_normalized[i, j] > thresh else "black", fontsize=6)

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Confusion matrix saved to {save_path}')





"""
###### version 5:    (Works)
def plot_confusion_matrix(label_path, pred_path, save_path='confusion_matrix.png', class_names=None):
    labels = np.loadtxt(label_path, dtype=int)
    preds = np.loadtxt(pred_path)
    preds = np.argmax(preds, axis=1)
    
    cm = confusion_matrix(labels, preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Generate class names 1–60 if not provided
    
    if class_names is None:
        class_names = [str(i) for i in range(1, cm.shape[0] + 1)]


    
    # Adjust figure size based on number of classes
    fig_width = max(30, len(class_names) * 0.35)
    fig_height = max(30, len(class_names) * 0.35)

    plt.figure(figsize=(fig_width, fig_height))
    df_cm = pd.DataFrame(cm_normalized, index=class_names, columns=class_names)
    
    sns.set(font_scale=1.2)  # Larger font for annotations
    sns.heatmap(df_cm, annot=True, fmt=".2f", cmap="Blues", cbar=True, 
                linewidths=0.5, linecolor='gray', square=True, annot_kws={"size": 10})

    plt.ylabel('True Label', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=16)
    plt.title('Normalized Confusion Matrix - NTU RGB+D 60', fontsize=18)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f'Confusion matrix saved to {save_path}')
"""



"""
### plot_confusion_matrix() version 4 works without the class_names:
def plot_confusion_matrix(label_path, pred_path, save_path='confusion_matrix.png', class_names=None):
    labels = np.loadtxt(label_path, dtype=int)
    preds = np.loadtxt(pred_path)
    preds = np.argmax(preds, axis=1)

    cm = confusion_matrix(labels, preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Adjust figure size based on number of classes
    fig_width = max(30, len(class_names) * 0.35)
    fig_height = max(30, len(class_names) * 0.35)

    plt.figure(figsize=(fig_width, fig_height))
    df_cm = pd.DataFrame(cm_normalized, index=class_names, columns=class_names)
    
    sns.set(font_scale=1.2)  # Larger font for annotations
    sns.heatmap(df_cm, annot=True, fmt=".2f", cmap="Blues", cbar=True, 
                linewidths=0.5, linecolor='gray', square=True, annot_kws={"size": 10})

    plt.ylabel('True Label', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=16)
    plt.title('Normalized Confusion Matrix - NTU RGB+D 60', fontsize=18)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f'Confusion matrix saved to {save_path}')
"""

""" ### plot_confusion_matrix() version 3:
def plot_confusion_matrix(label_path, pred_path, save_path='confusion_matrix.png', class_names=None):
    labels = np.loadtxt(label_path, dtype=int)
    preds = np.loadtxt(pred_path)
    preds = np.argmax(preds, axis=1)

    cm = confusion_matrix(labels, preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(20, 18))  # Bigger figure
    df_cm = pd.DataFrame(cm_normalized, index=class_names, columns=class_names)
    sns.set(font_scale=1.0)  # Control font size
    sns.heatmap(df_cm, annot=True, fmt=".2f", cmap="Blues", cbar=True, linewidths=0.5, linecolor='gray')
    
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.title('Normalized Confusion Matrix - NTU RGB+D 60', fontsize=16)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)  # High resolution
    print(f'Confusion matrix saved to {save_path}')
"""



"""### plot_confusion_matrix() version 1:
def plot_confusion_matrix(lable_path, pred_path, save_path='confusion_matrix.png', class_names=None):
    labels = np.loadtxt(lable_path, dtype=int)
    preds = np.loadtxt(pred_path)
    preds = np.argmax(preds, axis=1)

    cm = confusion_matrix(labels, preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    df_cm = pd.DataFrame(cm_normalized, index=class_names, columns=class_names)
    sns.heatmap(df_cm, annot=True, fmt=".2f", cmap="Blues", cbar=True)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f'Confusion matrix saved to {save_path}')
"""

""" #### Version 2 when plotting the confuison matrix: (works but not that good at visuals)
def plot_confusion_matrix(lable_path, pred_path, save_path='confusion_matrix.png', class_names=None):
    labels = np.loadtxt(lable_path, dtype=int)
    preds = np.loadtxt(pred_path)
    preds = np.argmax(preds, axis=1)

    cm = confusion_matrix(labels, preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    df_cm = pd.DataFrame(cm_normalized, index=class_names, columns=class_names)
    fig = sns.heatmap(df_cm, annot=True, fmt=".2f", cmap="Blues", cbar=True).get_figure()
    fig.savefig(save_path, bbox_inches='tight')
    print(f'Confusion matrix saved to {save_path}')
"""

######################################################################

def accuracy(output, target):
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = correct.view(-1).float().sum(0, keepdim=True)

    return correct.mul_(100.0 / batch_size)

def save_checkpoint(state, filename='checkpoint.pth.tar', is_best=False):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


################ performance calculations:

def measure_inference_time(model, inputs, device='cuda', repetitions=100):
    model = model.to(device)
    model.eval()
    inputs = inputs.to(device)

    if device == 'cuda':
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        timings = []
        with torch.no_grad():
            for _ in range(repetitions):
                starter.record()
                _ = model(inputs)
                ender.record()
                torch.cuda.synchronize()
                timings.append(starter.elapsed_time(ender))  # ms
        avg_time = sum(timings) / repetitions
    else:
        timings = []
        with torch.no_grad():
            for _ in range(repetitions):
                start = time.time()
                _ = model(inputs)
                end = time.time()
                timings.append((end - start) * 1000)  # ms
        avg_time = sum(timings) / repetitions

    print(f"Average inference time on {device.upper()}: {avg_time:.2f} ms")


if __name__ == '__main__':
    #main()
    
    ###### performances calculation:
    
    from fvcore.nn import FlopCountAnalysis, parameter_count
    
    bs, c, t, v = 64, 3, 20, 25
    
    
    parser = argparse.ArgumentParser(description='Skeleton-Based Action Recgnition')
    fit.add_fit_args(parser)
    parser.set_defaults(
    network ='ImpSGNv2', 
    dataset = 'NTU', #'NTU120',
    case = 0,
    batch_size=bs,
    max_epochs=140,
    monitor='val_acc',
    lr=0.001,
    weight_decay=0.0001,
    lr_factor=0.1,
    workers=16,
    print_freq = 20,
    train = 1,
    seg = 20,
    )
    args = parser.parse_args()


    args.num_classes = get_num_classes(args.dataset)
    model = ImpSGNv2(args.num_classes, args.dataset, args.seg, args)
    
    inputs = torch.randn(bs, t, c*v)
    
    ############ FLOPs + num params:
    flops = FlopCountAnalysis(model.cuda(), inputs.cuda())
    params = parameter_count(model)

    print(f"FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
    print(f"Params: {params[''] / 1e6:.2f} M")
    
    #############  Inference times (GPU / CPU):
    
    measure_inference_time(model, inputs, device='cuda')
    #measure_inference_time(model, inputs.cpu(), device='cpu')

    ############# Disk size: 
    torch.save(model.state_dict(), "temp_model.pth")
    size_mb = os.path.getsize("temp_model.pth") / 1e6
    print(f"Model size: {size_mb:.2f} MB")
    os.remove("temp_model.pth")
    
    ############## Peak memory usage (GPU):
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(inputs)
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"Peak memory usage: {peak_mem:.2f} MB")

    ############# Throughput:
    batch_size = bs
    start = time.time()
    for _ in range(10):
        _ = model(inputs)
    torch.cuda.synchronize()
    end = time.time()
    throughput = (10 * batch_size) / (end - start)
    print(f"Throughput: {throughput:.2f} samples/sec")





