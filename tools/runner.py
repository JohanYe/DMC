import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import json
import numpy as np
import warnings
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from pytorch3d.loss import chamfer_distance
from SAP.src.model import PSR2Mesh
from collections import defaultdict
from SAP.src.utils import *
import trimesh
from datetime import datetime
from tqdm import tqdm

# Suppress pkg_resources deprecation warnings from torch
warnings.filterwarnings(
    "ignore", category=UserWarning, module="torch.utils.cpp_extension"
)
# Suppress torch.meshgrid indexing argument warning
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")


def calibrate_batchnorm(model, dataloader, num_batches=100):
    model.train()  # BatchNorm needs train mode to update running stats
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if idx >= num_batches:
                break
            # Just forward pass, no backward
            (
                _,
                _,
                data,
                data_partial,
                value_centroid,
                value_std_pc,
                _,
                min_gt,
                max_gt,
            ) = batch
            partial = data_partial.cuda()
            _ = model(partial, min_gt, max_gt, value_std_pc, value_centroid)


def run_net(args, config, config_SAP, train_writer=None, val_writer=None):
    print("Training Start.......")
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(
        args, config.dataset.train
    ), builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # from IPython import embed; embed()

    # parameter setting
    start_epoch = 0
    best_metrics = None
    metrics = None

    # resume ckpts
    if args.resume:
        start_epoch, best_metrics_dict = builder.resume_model(
            base_model, args, logger=logger
        )
        if best_metrics_dict is not None:
            best_metrics = Metrics(config.consider_metric, best_metrics_dict)
        else:
            best_metrics = None
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger=logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log("Using Synchronized BatchNorm ...", logger=logger)
        base_model = nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[args.local_rank % torch.cuda.device_count()],
            find_unused_parameters=True,
        )
        print_log("Using Distributed Data parallel ...", logger=logger)
    else:
        print_log("Using Data parallel ...", logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()
    # -------
    # model_sap = Encode2Points(config_SAP).cuda()
    # load model
    """
    try:
        # load model
        state_dict = torch.load(os.path.join(config_SAP['train']['out_dir'], 'model.pt'))
        load_model_manual(state_dict['state_dict'], model)

        out = "Load model from iteration %d" % state_dict.get('it', 0)
        logger.info(out)
        # load point cloud
    except:
        state_dict = dict()
    """
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)

    # trainval
    # training
    base_model.zero_grad()
    criterion = nn.MSELoss()
    for epoch in range(start_epoch, config.max_epoch + 1):
        print("epoch:", epoch)
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_each = {}
        num_iter = 0
        loss = 0
        base_model.train()
        n_batches = len(train_dataloader)

        pbar = tqdm(enumerate(train_dataloader), total=n_batches, desc=f"Epoch {epoch}")

        for idx, (
            _,
            _,
            data,
            data_partial,
            value_centroid,
            value_std_pc,
            shell_grid_gt,
            min_gt,
            max_gt,
        ) in pbar:
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train._base_.N_POINTS
            dataset_name = config.dataset.train._base_.NAME

            if dataset_name == "crown":
                gt = data.cuda()
                partial = data_partial.cuda()
                gt_psr = shell_grid_gt.cuda()
            else:
                raise NotImplementedError(f"Train phase do not support {dataset_name}")

            num_iter += 1

            psr_grid, point_r = base_model(
                partial, min_gt, max_gt, value_std_pc, value_centroid
            )
            psr_grid = torch.tanh(torch.clamp(psr_grid, -10, 10))
            gt_psr = torch.tanh(torch.clamp(gt_psr, -10, 10))

            loss_chamfer = base_model.module.get_loss(point_r, gt)
            loss_mse = criterion(psr_grid, gt_psr)
            loss = loss_mse + loss_chamfer

            # Save point cloud for visualization
            if idx == 0:
                vis_dir = os.path.join(args.experiment_path, "vis")
                os.makedirs(vis_dir, exist_ok=True)
                pred_points = (
                    point_r[0].detach().cpu().numpy() * value_std_pc.numpy()
                ) + value_centroid.numpy()
                trimesh.PointCloud(pred_points).export(
                    f"./vis/epoch{epoch:03d}_train_sample{idx}_pred.ply"
                )

            # Update progress bar with individual losses
            pbar.set_postfix(
                {
                    "loss_mse": loss_mse.item(),
                    "loss_chamfer": loss_chamfer.item(),
                    "total_loss": loss.item(),
                }
            )

            if num_iter == config.step_per_update:
                num_iter = 0
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=1.0)
                optimizer.step()
                base_model.zero_grad()

            n_itr = epoch * n_batches + idx
            if train_writer is not None:
                train_writer.add_scalar("train/itr", loss, n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx == 5 and args.debug is True:
                break

        # Print per-epoch statistics
        if train_writer is not None:
            train_writer.add_scalar("train/epoch", loss, epoch)

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if epoch % args.val_freq == 0 and epoch != 0:
            print_log("Calibrating BatchNorm statistics...", logger=logger)
            calibrate_batchnorm(base_model.module, train_dataloader, num_batches=100)

            # Sync BatchNorm stats across processes if using SyncBatchNorm
            if args.distributed:
                torch.distributed.barrier()

            # Validate the current model
            metrics = validate(
                base_model,
                criterion,
                test_dataloader,
                epoch,
                val_writer,
                args,
                config,
                config_SAP,
                logger=logger,
            )

            # Save checkpoints
            if best_metrics is None or metrics < best_metrics:
                best_metrics = metrics
                logger.info("New best model (loss %.4f)" % metrics)
                builder.save_checkpoint(
                    base_model,
                    optimizer,
                    epoch,
                    metrics,
                    best_metrics,
                    "ckpt-best",
                    args,
                    logger=logger,
                )
        builder.save_checkpoint(
            base_model,
            optimizer,
            epoch,
            metrics,
            best_metrics,
            "ckpt-last",
            args,
            logger=logger,
        )
        if (config.max_epoch - epoch) < 10:
            builder.save_checkpoint(
                base_model,
                optimizer,
                epoch,
                metrics,
                best_metrics,
                f"ckpt-epoch-{epoch:03d}",
                args,
                logger=logger,
            )

    # Only close writers on rank 0
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


def validate(
    base_model,
    criterion,
    test_dataloader,
    epoch,
    val_writer,
    args,
    config,
    config_SAP,
    logger=None,
):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger=logger)
    base_model.eval()
    eval_list = defaultdict(list)
    eval_step_dict = {}
    eval_dict = {}
    category_metrics = dict()
    n_samples = len(test_dataloader)

    with torch.no_grad():
        pbar = tqdm(
            enumerate(test_dataloader),
            total=n_samples,
            desc=f"Validating Epoch {epoch}",
        )

        for idx, (
            taxonomy_ids,
            model_ids,
            data,
            data_partial,
            value_centroid,
            value_std_pc,
            shell_grid_gt,
            min_gt,
            max_gt,
        ) in pbar:
            taxonomy_id = (
                taxonomy_ids[0]
                if isinstance(taxonomy_ids[0], str)
                else taxonomy_ids[0].item()
            )
            model_id = model_ids[0]
            npoints = config.dataset.val._base_.N_POINTS
            dataset_name = config.dataset.val._base_.NAME
            if dataset_name == "crown":
                gt = data.cuda()
                partial = data_partial.cuda()
                gt_psr = shell_grid_gt.cuda()
            else:
                raise NotImplementedError(f"Train phase do not support {dataset_name}")

            psr_grid, point_r = base_model(
                partial, min_gt, max_gt, value_std_pc, value_centroid
            )

            # Save point cloud for visualization
            if idx == 0:
                vis_dir = os.path.join(args.experiment_path, "vis")
                os.makedirs(vis_dir, exist_ok=True)
                pred_points = (
                    point_r[0].cpu().numpy() * value_std_pc.numpy()
                ) + value_centroid.numpy()
                trimesh.PointCloud(pred_points).export(
                    f"./vis/epoch{epoch:03d}_validation_sample{idx}_pred.ply"
                )

            psr_grid = torch.tanh(torch.clamp(psr_grid, -10, 10))
            gt_psr = torch.tanh(torch.clamp(gt_psr, -10, 10))
            loss_chamfer = base_model.module.get_loss(point_r, gt)
            loss_mse = criterion(psr_grid, gt_psr)
            loss = loss_mse + loss_chamfer

            eval_step_dict["psr_l1"] = loss_chamfer.cpu().item()
            eval_step_dict["psr_l2"] = loss_mse.cpu().item()
            for k, v in eval_step_dict.items():
                eval_list[k].append(v)

            eval_dict = {k: np.mean(v) for k, v in eval_list.items()}

            # Update progress bar with current losses
            pbar.set_postfix(
                {"loss_mse": loss_mse.item(), "loss_chamfer": loss_chamfer.item()}
            )

            if val_writer is not None:
                val_writer.add_scalar(
                    "Valid/loss_chamfer", loss_chamfer.item(), epoch * n_samples + idx
                )
                val_writer.add_scalar(
                    "Valid/loss_mse", loss_mse.item(), epoch * n_samples + idx
                )
                val_writer.add_scalar(
                    "Valid/total_loss", loss.item(), epoch * n_samples + idx
                )

            if idx < 3:
                vis_dir = os.path.join(args.experiment_path, "vis")
                os.makedirs(vis_dir, exist_ok=True)

                pred_points = (
                    point_r[0].cpu().numpy() * value_std_pc.numpy()
                ) + value_centroid.numpy()
                gt_points = (
                    gt[0].cpu().numpy() * value_std_pc.numpy()
                ) + value_centroid.numpy()

                trimesh.PointCloud(pred_points).export(
                    os.path.join(vis_dir, f"epoch{epoch:03d}_sample{idx}_pred.ply")
                )
                trimesh.PointCloud(gt_points).export(
                    os.path.join(vis_dir, f"epoch{epoch:03d}_sample{idx}_gt.ply")
                )

            if idx == 5 and args.debug is True:
                break

    # Print per-epoch validation statistics
    print_log(
        f"[VALIDATION] Epoch {epoch} - MSE: {eval_dict['psr_l2']:.4f}, L1: {eval_dict['psr_l1']:.4f}",
        logger=logger,
    )

    if val_writer is not None:
        val_writer.add_scalar("Valid/epoch_loss_mse", eval_dict["psr_l2"], epoch)
        val_writer.add_scalar("Valid/epoch_loss_chamfer", eval_dict["psr_l1"], epoch)

    return eval_dict["psr_l2"]


def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log("Tester start ... ", logger=logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger=logger)
    # model_sap = Encode2Points(config_SAP).cuda()
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP
    if args.distributed:
        raise NotImplementedError()

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    test(
        base_model,
        test_dataloader,
        ChamferDisL1,
        ChamferDisL2,
        args,
        config,
        logger=logger,
    )


def test(
    base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger=None
):
    base_model.eval()  # set model to eval mode
    # use_cuda = not args.no_cuda and torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")
    # generator = get_generator(model_sap, config_SAP, device=device)
    # dpsr = DPSR(res=(128, 128, 128), sig=2)
    # test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    threshold = 0
    eval_list = defaultdict(list)
    eval_step_dict = {}
    eval_dict = {}
    # test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader)  # bs is 1
    print("Generating...")
    with torch.no_grad():
        for idx, (
            taxonomy_ids,
            model_ids,
            data,
            data_partial,
            value_centroid,
            value_std_pc,
            shell_grid_gt,
            min_gt,
            max_gt,
        ) in enumerate(test_dataloader):
            taxonomy_id = (
                taxonomy_ids[0]
                if isinstance(taxonomy_ids[0], str)
                else taxonomy_ids[0].item()
            )
            model_id = model_ids[0]

            npoints = config.dataset.test._base_.N_POINTS
            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == "crown":
                print("do test")
                gt = data.cuda()
                partial = data_partial.cuda()
                gt_psr = shell_grid_gt.cuda()
                min_gt = -4.21
                max_gt = 25.5

                min_gt = torch.from_numpy(np.asarray(min_gt)).float()
                max_gt = torch.from_numpy(np.asarray(max_gt)).float()
                # calculate mian and max
                psr_grid, points, point_r, min_depoint, max_depoint = base_model(
                    partial,
                    min_gt.cuda(),
                    max_gt.cuda(),
                    value_std_pc.cuda(),
                    value_centroid.cuda(),
                )
                point_dir = "./Results-pointr"
                dense_points = point_r
                dense_points = (
                    torch.multiply(dense_points[0].cpu(), value_std_pc) + value_centroid
                )
                np.save(
                    os.path.join(point_dir, str(model_id) + "pred.npy"),
                    dense_points.cpu().numpy(),
                )
                pc = trimesh.PointCloud(dense_points.cpu().numpy())
                pc.export(os.path.join(point_dir, str(model_id) + "pred.ply"))

                min_gt = dense_points.min()
                max_gt = dense_points.max()
                # min_gt = torch.from_numpy(np.asarray(min_gt)).float()
                # max_gt = torch.from_numpy(np.asarray(max_gt)).float()
                # now calculate mesh crown
                psr_grid, points, point_r, min_depoint, max_depoint = base_model(
                    partial,
                    min_gt.cuda(),
                    max_gt.cuda(),
                    value_std_pc.cuda(),
                    value_centroid.cuda(),
                )
                v, f, _ = mc_from_psr(psr_grid, zero_level=threshold)
                min_depoint = min_depoint.cpu().numpy()
                max_depoint = max_depoint.cpu().numpy()

                # denormalize
                de_p = v * (max_depoint + 1 - min_depoint) + min_depoint
                de_point = (
                    points.cpu().numpy() * (max_depoint + 1 - min_depoint) + min_depoint
                )

                # Write output
                mesh_dir = "./af"
                mesh_out_file = os.path.join(mesh_dir, str(model_id) + ".ply")
                export_mesh(mesh_out_file, de_p, f)
                # write point cloud

                np.save(
                    os.path.join(point_dir, str(model_id) + "predsap.npy"), de_point
                )
                loss_chamfer_l1 = ChamferDisL1(point_r, gt)
                loss_chamfer_l2 = ChamferDisL2(point_r, gt)
                # loss_chamfer = base_model.get_loss(point_r, gt)
                logger.info("Loss chamfer L1 : %.4f" % (loss_chamfer_l1))
                logger.info("Loss chamfer L2 : %.4f" % (loss_chamfer_l2))

                eval_step_dict["psr_l1"] = F.l1_loss(psr_grid, gt_psr).item()
                eval_step_dict["psr_l2"] = F.mse_loss(psr_grid, gt_psr).item()
                for k, v in eval_step_dict.items():
                    eval_list[k].append(v)

                eval_dict = {k: np.mean(v) for k, v in eval_list.items()}
                test_loss = eval_dict
                for k, v in eval_dict.items():
                    logger.info("test metric MSE : %.4f" % (test_loss["psr_l2"]))
                    logger.info("test metric L1 : %.4f" % (test_loss["psr_l1"]))

                continue
            else:
                raise NotImplementedError(f"Train phase do not support {dataset_name}")

            if (idx + 1) % 2 == 0:
                print_log(
                    "Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s"
                    % (
                        idx + 1,
                        n_samples,
                        taxonomy_id,
                        model_id,
                        ["%.4f" % l for k, v in eval_dict.items()],
                    ),
                    logger=logger,
                )
    return
