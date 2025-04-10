# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable

import torch
import numpy as np
import torch.nn.functional as F

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    kl_weight = 1e-3

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (points, labels, surface, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
#        print(data_iter_step)
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        surface = surface.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=False):
            outputs = model(surface, points)
            if 'kl' in outputs:
                loss_kl = outputs['kl']
                loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]
            else:
                loss_kl = None

            outputs = outputs['logits']


            loss = criterion(outputs, labels)

            
            if loss_kl is not None:
                loss = loss + kl_weight * loss_kl

        loss_value = loss.item()

        threshold = 0.5

        pred = torch.zeros_like(outputs)
        pred[torch.sigmoid(outputs)>=threshold] = 1

        intersection = (pred * labels).sum(dim=1)
        union = (pred + labels).gt(0).sum(dim=1) + 1e-5
        
        iou = intersection * 1.0 / union
        iou = iou.mean()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        metric_logger.update(loss_surf=loss.item())
        metric_logger.update(loss_near=loss.item())

        if loss_kl is not None:
            metric_logger.update(loss_kl=loss_kl.item())

        metric_logger.update(iou=iou.item())

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

        if torch.rand((1,)).item() > 0.99:
            import matplotlib.pyplot as plt

            # Assuming the surface is a 3D grid and outputs are the predicted SDF values
            points_np = points[0].detach().cpu().numpy()
            outputs_np = outputs[0].detach().cpu().numpy()
            print("points_np", points_np.shape)
            print("outputs_np", outputs_np.shape)

            # Create a 3D plot for the zero-level set
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Extract the zero-level set (isosurface where SDF = 0)
            zero_level_set = np.abs(outputs_np) < 1e-3
            xyz = points_np[zero_level_set]

            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='r', marker='o', s=1)
            ax.set_title("Zero-Level Set")
            plt.savefig("zero-level-set-train.png")
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.MSELoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for points, labels, surface, _ in metric_logger.log_every(data_loader, 50, header):

        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        surface = surface.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=False):

            outputs = model(surface, points)
            if 'kl' in outputs:
                loss_kl = outputs['kl']
                loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]
            else:
                loss_kl = None

            outputs = outputs['logits']

            loss = criterion(outputs, labels)

        threshold = 0

        pred = torch.zeros_like(outputs)
        pred[outputs>=threshold] = 1

        accuracy = (pred==labels).float().sum(dim=1) / labels.shape[1]
        accuracy = accuracy.mean()
        intersection = (pred * labels).sum(dim=1)
        union = (pred + labels).gt(0).sum(dim=1)
        iou = intersection * 1.0 / union + 1e-5
        iou = iou.mean()

        batch_size = points.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['iou'].update(iou.item(), n=batch_size)

        if torch.rand((1,)).item() > 0.98 and batch_size > 0:
            import matplotlib.pyplot as plt

            # Assuming the surface is a 3D grid and outputs are the predicted SDF values
            points_np = points[0].detach().cpu().numpy()
            outputs_np = outputs[0].detach().cpu().numpy()

            # Create a 3D plot for the zero-level set
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Extract the zero-level set (isosurface where UDF = 0)
            zero_level_set = np.abs(outputs_np) < 1e-3
            xyz = points_np[zero_level_set]

            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='r', marker='o', s=1)
            ax.set_title("Zero-Level Set")
            plt.savefig("zero-level-set-test.png")

        if loss_kl is not None:
            metric_logger.update(loss_kl=loss_kl.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* iou {iou.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(iou=metric_logger.iou, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
