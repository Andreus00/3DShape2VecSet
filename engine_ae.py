# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import os
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

PLOT = False

if PLOT:
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(161, projection='3d')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    ax2 = fig.add_subplot(162, projection='3d')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    ax3 = fig.add_subplot(163, projection='3d')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.legend()
    ax4 = fig.add_subplot(164, projection='3d')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.legend()
    ax5 = fig.add_subplot(165, projection='3d')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_zlabel('Z')
    ax5.legend()
    ax6 = fig.add_subplot(166, projection='3d')
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    ax6.set_zlabel('Z')
    ax6.legend()

    plt.title('3D Point Cloud with Labels as Color')

    plt.ion()
    plt.show()

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, args=None):
    
    if PLOT:
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
    grad_weight = 1e-4

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

        grads_mask = torch.bitwise_and(udf < args.max_dist*0.9, udf > 0.001).reshape(*gt_grads.shape[:2])
        
        n_queries = points.shape[1]

        with torch.cuda.amp.autocast(enabled=False):
            outputs = model(surface, points, with_grads=True)

            # KL loss
            if 'kl' in outputs and outputs['kl'] is not None:
                loss_kl = outputs['kl']
                loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]
            else:
                loss_kl = None
            
            # Gradients loss
            if 'grads' in outputs:
                grads = outputs['grads']
                loss_grads = (1 - F.cosine_similarity(grads[grads_mask], gt_grads[grads_mask], dim=-1)).mean()   # cosine distance (1-cos_sim)
                # loss_grads = F.mse_loss(F.normalize(grads, dim=-1), F.normalize(gt_grads, dim=-1))
            else:
                loss_grads = None

            # Point-wise Loss
            logits = outputs['logits']

            loss_near = criterion(logits[:, :n_near_pts], labels[:, :n_near_pts])
            loss_rand = criterion(logits[:, n_near_pts:n_near_pts+n_rnd_pts], labels[:, n_near_pts:n_near_pts+n_rnd_pts])
            loss_srf = criterion(logits[:, n_near_pts:n_near_pts+n_rnd_pts:], labels[:, n_near_pts:n_near_pts+n_rnd_pts:])

            # if PLOT:
            #     print('training loop > n_near_pts', n_near_pts, 'n_rnd_pts', n_rnd_pts, 'n_sfc_pts', n_sfc_pts)
            #     # Get near, random, and surface points
            #     near_points = points[0, :n_near_pts].cpu().detach().numpy()
            #     rand_points = points[0, n_near_pts:n_near_pts+n_rnd_pts].cpu().detach().numpy()
            #     srf_points = points[0, n_near_pts + n_rnd_pts:].cpu().detach().numpy()

            #     ax1.cla()
            #     ax1.scatter(near_points[:, 0], near_points[:, 1], near_points[:, 2], c='blue', s=1, label='Near Points')
            #     ax1.set_title('Near Points')
            #     ax1.legend()

            #     ax2.cla()
            #     ax2.scatter(rand_points[:, 0], rand_points[:, 1], rand_points[:, 2], c='green', s=1, label='Random Points')
            #     ax2.set_title('Random Points')
            #     ax2.legend()
            #     print(labels[:, n_near_pts:n_near_pts+n_rnd_pts])

            #     ax3.cla()
            #     ax3.scatter(srf_points[:, 0], srf_points[:, 1], srf_points[:, 2], c='red', s=1, label='Surface Points')
            #     ax3.set_title('Surface Points')
            #     ax3.legend()
            #     plt.draw()
            #     plt.pause(100)

            loss = loss_near + loss_rand + loss_srf

            if data_iter_step == 0:

                if PLOT:
                    # Pick 10,000 random points
                    sampled_points = points[0].cpu().detach().numpy()
                    sampled_labels = logits[0].detach().cpu().numpy() #torch.sigmoid(logits[0, idxs]).cpu().detach().numpy()
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
                    sampled_labels = (logits[0].flatten() - labels[0].flatten()).abs().detach().cpu().numpy() #torch.sigmoid(logits[0, idxs]).cpu().detach().numpy()
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
                    sampled_labels = labels[0].detach().cpu().numpy() #torch.sigmoid(logits[0, idxs]).cpu().detach().numpy()
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

                    if 'grads' in outputs:
                        sampled_points = points[grads_mask].reshape(-1, 3)
                        pred_grads = grads[grads_mask].detach().cpu().numpy()
                        gt_grads_np = gt_grads[grads_mask].detach().cpu().numpy()
                        gt_udf = labels[grads_mask].detach().cpu().numpy()
                        pred_udf = logits[grads_mask].detach().cpu().numpy()
                        cos_dist_vals = (1 - F.cosine_similarity(grads[grads_mask], gt_grads[grads_mask], dim=-1))
                        cos_dist_vals_np = cos_dist_vals.detach().cpu().numpy()
                        sorted_indices = np.argsort(cos_dist_vals_np)
                        num_grad_samples = min(1000, sampled_points.shape[0])
                        
                        lowest_indices = sorted_indices[:num_grad_samples]
                        highest_indices = sorted_indices[-num_grad_samples:]
                        sample_idxs = np.random.choice(sampled_points.shape[0], num_grad_samples, replace=False)
                        # pred_grads = pred_grads[sample_idxs]
                        # gt_grads_np = gt_grads_np[sample_idxs]
                        # pred_grads = pred_grads[lowest_indices]
                        # gt_grads_np = gt_grads_np[lowest_indices]
                        # pred_grads = pred_grads[highest_indices]
                        # gt_grads_np = gt_grads_np[highest_indices]
                        # gt_udf_highest = gt_udf[highest_indices]
                        # print(gt_udf_highest)


                        log_n, lab_n, gt_grad_n = logits[:, :n_near_pts], labels[:, :n_near_pts], gt_grads[:, :n_near_pts]
                        log_r, lab_r, gt_grad_r = logits[:, n_near_pts:n_near_pts+n_rnd_pts], labels[:, n_near_pts:n_near_pts+n_rnd_pts], gt_grads[:, n_near_pts:n_near_pts+n_rnd_pts]
                        log_s, lab_s, gt_grad_s = logits[:, n_near_pts:n_near_pts+n_rnd_pts:], labels[:, n_near_pts:n_near_pts+n_rnd_pts:], gt_grads[:, n_near_pts:n_near_pts+n_rnd_pts:]

                        # err_n = torch.nn.functional.mse_loss(log_n, lab_n, dim=-1)
                        # err_r = torch.nn.functional.mse_loss(log_r, lab_r, dim=-1)
                        # err_s = torch.nn.functional.mse_loss(log_s, lab_s, dim=-1)
                        err = (logits - labels)[grads_mask].detach().cpu()
                        # Create a color vector for gt_udf: blue for near, green for rnd, red for srf
                        color_vec = np.zeros((gt_udf.shape[0], 3))
                        # Indices for each set
                        near_idx = np.arange(n_near_pts)
                        rnd_idx = np.arange(n_near_pts, n_near_pts + n_rnd_pts)
                        srf_idx = np.arange(n_near_pts + n_rnd_pts, n_near_pts + n_rnd_pts + n_sfc_pts)
                        # Mask for grads_mask
                        mask_indices = np.where(grads_mask.detach().cpu().flatten())[0]
                        # Map set indices to mask_indices
                        near_mask = np.isin(mask_indices, near_idx)
                        rnd_mask = np.isin(mask_indices, rnd_idx)
                        srf_mask = np.isin(mask_indices, srf_idx)
                        # Assign colors
                        color_vec[near_mask] = [0, 0, 1]   # blue
                        color_vec[rnd_mask] = [0, 1, 0]    # green
                        color_vec[srf_mask] = [1, 0, 0]    # red

                        sampled_points = sampled_points.cpu().detach().numpy() # [highest_indices]

                        # Plot error (cos_dist_vals_np) vs gt_udf_highest as a scatter plot
                        ax4.set_title("Cosine Distance vs UDF (Surface Proximity)")
                        ax4.set_xlabel("GT UDF (Distance to Surface)")
                        ax4.set_ylabel("Cosine Distance (Error)")
                        # Clear previous 2D plot if any
                        if hasattr(ax4, '_error_scatter'):
                            ax4._error_scatter.remove()
                        ax4._error_scatter = ax4.figure.add_axes([0.7, 0.1, 0.25, 0.25])
                        ax4._error_scatter.cla()

                        sc4 = ax4._error_scatter.scatter(gt_udf, cos_dist_vals_np, s=2, alpha=0.5, color=color_vec)
                        # sc4 = ax4._error_scatter.scatter(gt_udf, cos_dist_vals_np, s=2, alpha=0.5, c=err)
                        ax4._error_scatter.set_xlabel("GT UDF")
                        ax4._error_scatter.set_ylabel("Cosine Distance")
                        ax4._error_scatter.set_title("Error vs UDF")
                        ax4._error_scatter.grid(True)

                        # Plot a central slice of the predicted UDF vs GT UDF
                        # Assume points are in shape [N, 3], logits and labels are [N]
                        # We'll plot points where z is close to the median z (central slice)
                        z_vals = sampled_points[:, 2]
                        z_center = np.median(z_vals)
                        slice_thickness = 0.02  # adjust as needed
                        slice_mask = np.abs(z_vals - z_center) < slice_thickness

                        slice_points = sampled_points[slice_mask]
                        
                        slice_pred_udf = pred_udf[slice_mask]
                        slice_gt_udf = gt_udf[slice_mask]

                        sc_pred = ax5.scatter(slice_points[:, 0], slice_points[:, 1], c=slice_pred_udf, cmap='viridis', s=2)
                        ax5.set_title('Predicted UDF (central slice)')
                        ax5.set_xlabel('X')
                        ax5.set_ylabel('Y')

                        plt.tight_layout()

                        # Plot on ax6 some of the gradients whose error is high
                        # Select gradients with highest cosine distance (error)
                        num_grad_samples = min(200, sampled_points.shape[0])
                        high_error_indices = sorted_indices[-num_grad_samples:]

                        high_error_points = sampled_points[high_error_indices]
                        high_error_pred_grads = pred_grads[high_error_indices]
                        high_error_gt_grads = gt_grads_np[high_error_indices]
                        high_error_vals = cos_dist_vals_np[high_error_indices]

                        ax6.cla()
                        ax6.set_title('High Error Gradients')
                        ax6.set_xlabel('X')
                        ax6.set_ylabel('Y')
                        ax6.set_zlabel('Z')

                        # Plot points colored by error
                        sc6 = ax6.scatter(
                            high_error_points[:, 0],
                            high_error_points[:, 1],
                            high_error_points[:, 2],
                            c=high_error_vals,
                            cmap='hot',
                            s=8,
                            alpha=0.8,
                            label='High Error Points'
                        )

                        # Plot predicted gradients in blue
                        ax6.quiver(
                            high_error_points[:, 0], high_error_points[:, 1], high_error_points[:, 2],
                            high_error_pred_grads[:, 0], high_error_pred_grads[:, 1], high_error_pred_grads[:, 2],
                            color='blue', length=0.05, normalize=True, label='Predicted'
                        )
                        # Plot GT gradients in red
                        ax6.quiver(
                            high_error_points[:, 0], high_error_points[:, 1], high_error_points[:, 2],
                            high_error_gt_grads[:, 0], high_error_gt_grads[:, 1], high_error_gt_grads[:, 2],
                            color='red', length=0.05, normalize=True, label='GT'
                        )
                        ax6.quiver(
                            high_error_points[:, 0], high_error_points[:, 1], high_error_points[:, 2],
                            high_error_gt_grads[:, 0], high_error_gt_grads[:, 1], high_error_gt_grads[:, 2],
                            color='red', length=0.05, normalize=True, label='GT'
                        )

                        surface_points = surface[0].cpu().detach().numpy()
                        ax6.scatter(
                            surface_points[:, 0],
                            surface_points[:, 1],
                            surface_points[:, 2],
                            c='red',
                            label='Surface Points'
                        )
                        ax6.scatter(
                            high_error_points[:, 0], high_error_points[:, 1], high_error_points[:, 2],
                            color='blue', label='high_error_points'
                        )
                        # ax6.quiver(
                        #     high_error_points[:, 0], high_error_points[:, 1], high_error_points[:, 2],
                        #     high_error_gt_grads[:, 0], high_error_gt_grads[:, 1], high_error_gt_grads[:, 2],
                        #     color='red', length=0.05, normalize=True, label='GT'
                        # )

                        # Sample only a subset of the gradients for visualization

                        # ax4.quiver(
                        #     sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2],
                        #     pred_grads[:, 0], pred_grads[:, 1], pred_grads[:, 2],
                        #     color='blue', length=0.05, normalize=True, label='Predicted Gradients'
                        # )
                        # ax4.quiver(
                        #     sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2],
                        #     gt_grads_np[:, 0], gt_grads_np[:, 1], gt_grads_np[:, 2],
                        #     color='red', length=0.05, normalize=True, label='GT Gradients'
                        # )
                        # pts = points[grads_mask].detach().cpu().numpy()
                        
                        # sc4 = ax4.scatter(
                        #     pts[:, 0],
                        #     pts[:, 1],
                        #     pts[:, 2],
                        #     c=cos_dist.flatten().detach().cpu().numpy(),
                        #     label='Surface Points'
                        # )

                    
                    if epoch == 1 and data_iter_step == 0:
                        plt.colorbar(sc1, label='Labels')
                        plt.colorbar(sc2, label='Labels')
                        plt.colorbar(sc3, label='Labels')
                        plt.colorbar(sc4, label='Gradient Cos Dist')
                        plt.colorbar(sc6, ax=ax6, label='Cosine Distance (Error)')

                    plt.draw()
                    plt.pause(1.5)

                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    model = model.module
                latent = model(surface[:1], None, only_encode=True)[1]

                def callable_udf_func(x, udf_th):
                    with torch.no_grad():
                        x_nograd = x.clone().detach().to(device).unsqueeze(0)
                        udf = model(latent.detach(), x_nograd, only_decode=True)['logits'].flatten()

                    grad = torch.zeros_like(x, device=x.device)
                    mask = udf < udf_th

                    if mask.sum() > 0:
                        x_grad = x[mask].clone().detach().requires_grad_(True).unsqueeze(0)
                        udf_grad = model(latent.detach(), x_grad, only_decode=True)['logits'].flatten()
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
                
                if epoch > 1:
                    try:
                        verts, faces = get_mesh_from_udf(
                            udf_func=callable_udf_func,
                            coords_range=(-1, 1),
                            max_dist=0.1,
                            N=256,
                            use_fast_grid_filler=False,
                            th_alpha=1.05,
                            th_beta=1.75,
                            device=device
                        )

                        mesh = trimesh.Trimesh(vertices=verts.detach().cpu().numpy(), faces=faces.detach().cpu().numpy())
                        p = f'{args.output_dir}/final_{epoch}_{data_iter_step}.obj'
                        mesh.export(p, file_type='obj')
                        print(f"Mesh exported at {p}")
                    except Exception as e:
                        print(e)

            if loss_kl is not None:
                loss = loss + kl_weight * loss_kl
            if loss_grads is not None:
                loss = loss + grad_weight * loss_grads

        loss_value = loss.item()

        threshold = 0.5

        pred = torch.zeros_like(logits)
        pred[torch.sigmoid(logits)>=threshold] = 1

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
