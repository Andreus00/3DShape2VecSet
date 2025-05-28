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
import matplotlib.pyplot as plt
from custom_mc.meshudf import get_mesh_from_udf
import trimesh

fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(141, projection='3d')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.legend()
ax2 = fig.add_subplot(142, projection='3d')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.legend()
ax3 = fig.add_subplot(143, projection='3d')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
ax3.legend()
ax4 = fig.add_subplot(144, projection='3d')
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('Z')
ax4.legend()

plt.title('3D Point Cloud with Labels as Color')

plt.ion()
plt.show()

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, args=None):
    global ax1, ax2, ax3, ax4

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter


    n_rnd_pts = int(args.random_samples_ratio * args.num_samples)
    n_sfc_pts = int(args.surface_samples_ratio * args.num_samples)
    n_near_pts = args.num_samples - (n_rnd_pts + n_sfc_pts)

    optimizer.zero_grad()

    kl_weight = 1e-3
    grad_weight = 1e-3

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (points, udf, surface, gt_grads, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
#        print(data_iter_step)
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        points = points.to(device, non_blocking=True)
        udf = udf.to(device, non_blocking=True)
        labels = torch.clip(udf, 0, args.max_dist)
        surface = surface.to(device, non_blocking=True)
        gt_grads = gt_grads.to(device)
        
        n_queries = points.shape[1]

        with torch.cuda.amp.autocast(enabled=False):
            outputs = model(surface, points)

            # KL loss
            if 'kl' in outputs and outputs['kl'] is not None:
                loss_kl = outputs['kl']
                loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]
            else:
                loss_kl = None
            
            # Gradients loss
            if 'grads' in outputs:
                grads = outputs['grads']
                loss_grads = F.mse_loss(grads, gt_grads, reduce=True, reduction='mean')
            else:
                loss_grads = None

            # Point-wise Loss
            outputs = outputs['logits']

            loss_near = criterion(outputs[:, :n_near_pts], labels[:, :n_near_pts])
            loss_rand = criterion(outputs[:, n_near_pts:n_near_pts+n_rnd_pts], labels[:, n_near_pts:n_near_pts+n_rnd_pts])
            loss_srf = criterion(outputs[:, n_near_pts:n_near_pts+n_rnd_pts:], labels[:, n_near_pts:n_near_pts+n_rnd_pts:])

            loss = loss_near + loss_rand + loss_srf

            if epoch % 1 == 0 and data_iter_step == 0:
                # Pick 10,000 random points
                sampled_points = points[0].cpu().detach().numpy()
                sampled_labels = outputs[0].detach().cpu().numpy() #torch.sigmoid(outputs[0, idxs]).cpu().detach().numpy()
                # Plot in 3D using labels as color
                ax1.cla()
                # ax1 = fig.add_subplot(131, projection='3d')
                sc1 = ax1.scatter(
                    sampled_points[:, 0],
                    sampled_points[:, 1],
                    sampled_points[:, 2],
                    c=sampled_labels,
                    cmap='viridis',
                    s=1
                )
                # Plot surface points in red
                # surface_points = surface[0].cpu().detach().numpy()
                # ax1.scatter(
                #     surface_points[:, 0],
                #     surface_points[:, 1],
                #     surface_points[:, 2],
                #     c='red',
                #     label='Surface Points'
                # )
                sampled_points = points[0].cpu().detach().numpy()
                sampled_labels = (outputs[0].flatten() - labels[0].flatten()).abs().detach().cpu().numpy() #torch.sigmoid(outputs[0, idxs]).cpu().detach().numpy()
                ax2.cla()
                # ax2 = fig.add_subplot(132, projection='3d')
                sc2 = ax2.scatter(
                    sampled_points[:, 0],
                    sampled_points[:, 1],
                    sampled_points[:, 2],
                    c=sampled_labels,
                    cmap='viridis',
                    s=1
                )
                # Plot surface points in red
                # surface_points = surface[0].cpu().detach().numpy()
                # ax2.scatter(
                #     surface_points[:, 0],
                #     surface_points[:, 1],
                #     surface_points[:, 2],
                #     c='red',
                #     label='Random Points'
                # )

                sampled_points = points[0].cpu().detach().numpy()
                sampled_labels = labels[0].detach().cpu().numpy() #torch.sigmoid(outputs[0, idxs]).cpu().detach().numpy()
                ax3.cla()
                # ax3 = fig.add_subplot(133, projection='3d')
                sc3 = ax3.scatter(
                    sampled_points[:, 0],
                    sampled_points[:, 1],
                    sampled_points[:, 2],
                    c=sampled_labels,
                    cmap='viridis',
                    s=1
                )
                ax4.cla()
                # Plot predicted gradients in blue and gt gradients in red
                num_grad_samples = min(1000, sampled_points.shape[0])
                sample_idxs = np.random.choice(grads.shape[1], num_grad_samples, replace=False)
                pred_grads = grads[0].detach().cpu().numpy()[sample_idxs]
                gt_grads_np = gt_grads[0].detach().cpu().numpy()[sample_idxs]
                sampled_points = points[0].cpu().detach().numpy()[sample_idxs]
                # Sample only a subset of the gradients for visualization

                ax4.quiver(
                    sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2],
                    pred_grads[:, 0], pred_grads[:, 1], pred_grads[:, 2],
                    color='blue', length=0.05, normalize=True, label='Predicted Gradients'
                )
                ax4.quiver(
                    sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2],
                    gt_grads_np[:, 0], gt_grads_np[:, 1], gt_grads_np[:, 2],
                    color='red', length=0.05, normalize=True, label='GT Gradients'
                )
                
                if epoch == 1 and data_iter_step == 0:
                    plt.colorbar(sc1, label='Labels')
                    plt.colorbar(sc2, label='Labels')
                    plt.colorbar(sc3, label='Labels')
                plt.draw()
                plt.pause(1.5)

                latent = model.encode(surface[:1])[1]

                def callable_udf_func(x, udf_th):
                    with torch.no_grad():
                        x_nograd = x.clone().detach().to(device).unsqueeze(0)
                        udf = model.decode(latent.detach(), x_nograd).flatten()

                    grad = torch.zeros_like(x, device=x.device)
                    mask = udf < udf_th
                    print(mask.sum(), udf.min(), udf_th)

                    if mask.sum() > 0:
                        x_grad = x[mask].clone().detach().requires_grad_(True).unsqueeze(0)
                        udf_grad = model.decode(latent.detach(), x_grad).flatten()
                        grad_outputs = torch.ones_like(udf_grad)
                        grads = torch.autograd.grad(
                            outputs=udf_grad,
                            inputs=x_grad,
                            grad_outputs=grad_outputs,
                            create_graph=False,
                            retain_graph=False,
                            only_inputs=True,
                            allow_unused=True
                        )[0]
                        grad[mask] = grads
                    return udf, -grad.detach()
                
                try:
                    verts, faces = get_mesh_from_udf(
                        udf_func=callable_udf_func,
                        coords_range=(-1, 1),
                        max_dist=0.1,
                        N=64,
                        use_fast_grid_filler=False,
                        th_alpha=1.05,
                        th_beta=1.75
                    )

                    mesh = trimesh.Trimesh(vertices=verts.detach().cpu().numpy(), faces=faces.detach().cpu().numpy())
                    mesh.export(f'mesh_output/final_{epoch}_{data_iter_step}.obj', file_type='obj')
                    print(f"Mesh exported at mesh_output/final_{epoch}_{data_iter_step}.obj")
                except Exception as e:
                    print(e)

            if loss_kl is not None:
                loss = loss + kl_weight * loss_kl
            if loss_grads is not None:
                loss = loss + grad_weight * loss_grads

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

        # loss /= accum_iter
        # loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()


        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        metric_logger.update(loss_near=loss_near.item())
        metric_logger.update(loss_rand=loss_rand.item())
        metric_logger.update(loss_srf=loss_srf.item())

        if loss_kl is not None:
            metric_logger.update(loss_kl=loss_kl.item())
        if loss_grads is not None:
            metric_logger.update(loss_grads=loss_grads.item())

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

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, max_dist):
    criterion = torch.nn.BCEWithLogitsLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for points, udf, surface, gt_grads, _ in metric_logger.log_every(data_loader, 50, header):

        points = points.to(device, non_blocking=True)
        udf = udf.to(device, non_blocking=True)
        labels = torch.clip(udf, 0, max_dist)
        surface = surface.to(device, non_blocking=True)
        gt_grads = gt_grads.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=False):

            outputs = model(surface, points, with_grads=False)
            if 'kl' in outputs and outputs['kl'] is not None:
                loss_kl = outputs['kl']
                loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]
            else:
                loss_kl = None

            # Gradients loss
            if 'grads' in outputs:
                grads = outputs['grads']
                loss_grads = F.mse_loss(grads, gt_grads, reduce=True, reduction='mean')
            else:
                loss_grads = None

            outputs = outputs['logits']

            loss = criterion(outputs, labels)

        threshold = 0.5

        pred = torch.zeros_like(outputs)
        outputs = torch.sigmoid(outputs)
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

        if loss_kl is not None:
            metric_logger.update(loss_kl=loss_kl.item())

        if loss_grads is not None:
            metric_logger.update(loss_grads=loss_grads.item())


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* iou {iou.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(iou=metric_logger.iou, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
