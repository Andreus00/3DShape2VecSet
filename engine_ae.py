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
import trimesh.rendering

import util.misc as misc
import util.lr_sched as lr_sched
import matplotlib.pyplot as plt
from custom_mc.meshudf import get_mesh_from_udf
import trimesh

import io
from PIL import Image
import wandb

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

    kl_weight = args.kl_weight
    grad_weight = args.grad_weight

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    logging_dict = {"epoch": epoch}
    global_offset = [args.global_offset_x, args.global_offset_y, args.global_offset_z]
    global_offset = torch.tensor(global_offset, dtype=torch.float32, device=device)

    for data_iter_step, (points, udf, surface, gt_grads, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
#        print(data_iter_step)
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        points = points.to(device, non_blocking=True).to(torch.float16)
        udf = udf.to(device, non_blocking=True).to(torch.float16)
        surface = surface.to(device, non_blocking=True).to(torch.float16)
        udf = udf.to(torch.float16)

        # points = points + global_offset
        # surface = surface + global_offset

        # points = points * args.global_scale
        # surface = surface * args.global_scale
        # udf = udf * args.global_scale

        if args.mse_loss:
            labels = torch.clip(udf, 0, args.max_dist)
        else:
            labels = torch.clip(udf, 0, args.max_dist)
            labels = 1 - (labels / args.max_dist)
        gt_grads = gt_grads.to(device)

        grads_mask = torch.bitwise_and(udf < args.max_dist*0.9, udf > 0.0001).reshape(*gt_grads.shape[:2])
        

        with_grads = args.grad_weight > 0.0

        with torch.cuda.amp.autocast(enabled=False):
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                print(points, udf, surface, gt_grads)
                outputs = model(surface, points, with_grads=with_grads)

                # KL loss
                if 'kl' in outputs and outputs['kl'] is not None:
                    loss_kl = outputs['kl']
                    loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]
                else:
                    loss_kl = None
                
                # Gradients loss
                if 'grads' in outputs:
                    if args.mse_loss:
                        grads = outputs['grads']    # if mse_loss, grads are already in the right direction
                    else:
                        grads = -outputs['grads']   # if sigmoid loss, grads are in the opposite direction
                    loss_grads = (1 - F.cosine_similarity(grads[grads_mask], gt_grads[grads_mask], dim=-1)).mean()
                else:
                    loss_grads = None

                # Point-wise Loss
                logits = outputs['logits']

                if args.model == "hunyuan_garments":
                    loss_near = criterion(logits[:, :n_near_pts], labels[:, :n_near_pts])
                    loss_rand = criterion(logits[:, n_near_pts:], labels[:, n_near_pts:])
                    loss_srf = None
                    loss = loss_near + loss_rand
                else:
                    loss_near = criterion(logits[:, :n_near_pts], labels[:, :n_near_pts])
                    loss_rand = criterion(logits[:, n_near_pts:n_near_pts+n_rnd_pts], labels[:, n_near_pts:n_near_pts+n_rnd_pts])
                    loss_srf = criterion(logits[:, n_near_pts:n_near_pts+n_rnd_pts:], labels[:, n_near_pts:n_near_pts+n_rnd_pts:])

                    loss = loss_near + loss_rand + loss_srf

                if data_iter_step == 0 and not args.model == "hunyuan_garments":

                    if PLOT:
                        
                        sampled_points = points[0].cpu().detach().numpy()
                        if args.mse_loss:
                            sampled_labels = logits[0].detach().cpu().numpy()
                        else:
                            sampled_labels = torch.sigmoid(logits[0]).detach().cpu().numpy()
                        
                        ax1.cla()
                        sc1 = ax1.scatter(
                            sampled_points[:, 0],
                            sampled_points[:, 1],
                            sampled_points[:, 2],
                            c=sampled_labels,
                            cmap='viridis',
                            s=1
                        )

                        error_labels = np.abs(sampled_labels - labels[0].flatten().detach().cpu().numpy()) 
                        ax2.cla()
                        sc2 = ax2.scatter(
                            sampled_points[:, 0],
                            sampled_points[:, 1],
                            sampled_points[:, 2],
                            c=error_labels,
                            cmap='viridis',
                            s=1
                        )

                        true_labels = labels[0].detach().cpu().numpy()
                        ax3.cla()
                        sc3 = ax3.scatter(
                            sampled_points[:, 0],
                            sampled_points[:, 1],
                            sampled_points[:, 2],
                            c=true_labels,
                            cmap='viridis',
                            s=1
                        )
                        ax4.cla()

                        if 'grads' in outputs:
                            sampled_points = points[grads_mask].reshape(-1, 3)
                            pred_grads = grads[grads_mask].detach().cpu().numpy()
                            gt_grads_np = gt_grads[grads_mask].detach().cpu().numpy()
                            gt_udf = udf[grads_mask].detach().cpu().numpy()
                            if args.mse_loss:
                                pred_udf = logits[grads_mask].detach().cpu().numpy()
                            else:
                                pred_udf = ((1 - torch.sigmoid(logits[grads_mask])) * args.max_dist).detach().cpu().numpy()

                            cos_dist_vals = (1 - F.cosine_similarity(grads[grads_mask], gt_grads[grads_mask], dim=-1))
                            cos_dist_vals_np = cos_dist_vals.detach().cpu().numpy()
                            sorted_indices = np.argsort(cos_dist_vals_np)
                            num_grad_samples = min(1000, sampled_points.shape[0])

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

                            sc4 = ax4._error_scatter.scatter(gt_udf, cos_dist_vals_np, s=2, alpha=0.5, c='b')
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
                            if args.mse_loss:
                                udf = model(latent.detach(), x_nograd, only_decode=True)['logits'].flatten()
                            else:
                                udf = (1 - torch.nn.functional.sigmoid(model(latent.detach(), x_nograd, only_decode=True)['logits'].flatten())) * args.max_dist

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

                            if args.mse_loss: # invert grads if using mse loss as they are opposite to the surface
                                grads = -grads

                            grad[mask] = grads
                        
                        return udf, grad.detach()
                    
                
                    if epoch > 0 and data_iter_step == 0:
                        try:
                                coords_range = torch.asarray(((-1 + global_offset[0]) * args.global_scale, args.global_scale * (1 + global_offset[0])), device='cpu')
                                verts, faces = get_mesh_from_udf(
                                    udf_func=callable_udf_func,
                                    coords_range=coords_range,
                                    max_dist=0.1,
                                    N=256,
                                    use_fast_grid_filler=False,
                                    th_alpha=1.05,
                                    th_beta=1.75,
                                    device=device
                                )

                                mesh = trimesh.Trimesh(vertices=verts.detach().cpu().numpy(), faces=faces.detach().cpu().numpy())
                                p = f'{args.output_dir}_mesh/final_{epoch}_{data_iter_step}.obj'
                                if not os.path.exists(os.path.dirname(p)):
                                    os.makedirs(os.path.dirname(p))
                                mesh.export(p, file_type='obj')
                                print(f"Mesh exported at {p}")
                                scene = trimesh.Scene(mesh)
                                data = scene.save_image(resolution=(1080,1080))
                                image =Image.open(io.BytesIO(data))
                                if image.mode != 'RGB':
                                    image = image.convert('RGB')
                                logging_dict["renders"] = wandb.Image(image, caption=f"reconstructed mesh")
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

        # Log batch losses to wandb
        logging_dict["batch/loss"] = loss_value
        logging_dict["batch/loss_near"] = loss_near.item()
        logging_dict["batch/loss_rand"] = loss_rand.item()
        if loss_srf is not None:
            logging_dict["batch/loss_srf"] = loss_srf.item()
        if loss_kl is not None:
            logging_dict["batch/loss_kl"] = loss_kl.item()
        if loss_grads is not None:
            logging_dict["batch/loss_grads"] = loss_grads.item()
        logging_dict["batch/iou"] = iou.item()
        wandb.log(logging_dict)

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
    logging_dict.update({f"epoch/{k}":v.global_avg for k, v in metric_logger.meters.items()})
    wandb.log(logging_dict)
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
