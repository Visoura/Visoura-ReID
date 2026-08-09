import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            logger.info('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    gram_loss_meter = AverageMeter()
    distill_loss_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()

    # ── wandb helpers ─────────────────────────────────────────────────
    use_wandb = cfg.WANDB.ENABLE
    wandb_log_freq = cfg.WANDB.LOG_FREQ if use_wandb else 0
    if use_wandb:
        import wandb
    global_step = 0

    # ── Text alignment freeze schedule ────────────────────────────────
    use_text = bool(cfg.MODEL.TEXT_EMB_PATH)
    use_proj_head = cfg.MODEL.TEXT_PROJ_HEAD
    warmup_epochs = cfg.MODEL.TEXT_PROJ_WARMUP_EPOCHS if use_proj_head else 0

    def _get_raw_model(m):
        """Unwrap DDP if needed."""
        return m.module if hasattr(m, 'module') else m

    def _apply_freeze_schedule(epoch):
        """Phase 1: freeze backbone, train projector only.
           Phase 2: unfreeze everything with differential LRs."""
        raw = _get_raw_model(model)

        if epoch == 1 and use_text and use_proj_head and warmup_epochs > 0:
            # Phase 1: freeze all except text_projector
            for name, param in raw.named_parameters():
                if "text_projector" not in name:
                    param.requires_grad = False
            logger.info("Phase 1: backbone frozen, training text_projector only "
                        "(epochs 1-%d).", warmup_epochs)

        if epoch == warmup_epochs + 1 and use_text and use_proj_head and warmup_epochs > 0:
            # Phase 2: unfreeze everything
            for param in raw.parameters():
                param.requires_grad = True
            logger.info("Phase 2: full model unfrozen with differential LRs.")

            # Set differential LRs in-place on existing param groups
            for pg in optimizer.param_groups:
                if pg.get("is_backbone", True):
                    pg["lr"] = cfg.SOLVER.BASE_LR * 0.1
                else:
                    pg["lr"] = cfg.SOLVER.BASE_LR

    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        gram_loss_meter.reset()
        distill_loss_meter.reset()
        evaluator.reset()

        _apply_freeze_schedule(epoch)
        model.train()

        for n_iter, batch in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()

            # Unpack: could be 4, 5, 6, or 7 elements depending on text_emb and guided_attention
            text_emb = None
            mask = None
            img_paths = None
            if len(batch) == 7:
                img, vid, target_cam, target_view, img_paths, text_emb, mask = batch
                text_emb = text_emb.to(device)
            elif len(batch) == 6:
                if use_text:
                    img, vid, target_cam, target_view, img_paths, text_emb = batch
                    text_emb = text_emb.to(device)
                else:
                    img, vid, target_cam, target_view, img_paths, mask = batch
            elif len(batch) == 5:
                img, vid, target_cam, target_view, img_paths = batch
            else:
                img, vid, target_cam, target_view = batch

            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            if mask is not None:
                mask = mask.to(device)

            with amp.autocast(enabled=True):
                model_out = model(img, target, cam_label=target_cam, view_label=target_view, mask=mask, img_paths=img_paths)

                student_tokens = None
                teacher_tokens = None
                dist_output = None
                if isinstance(model_out, tuple):
                    if len(model_out) == 6:
                        score, feat, text_proj, student_tokens, teacher_tokens, dist_output = model_out
                    elif len(model_out) == 5:
                        score, feat, text_proj, student_tokens, teacher_tokens = model_out
                    elif len(model_out) == 3:
                        score, feat, text_proj = model_out
                    else:
                        score, feat = model_out
                        text_proj = None
                else:
                    score, feat = model_out
                    text_proj = None

                # vision_feat extraction for dist_loss is handled inside loss_fn
                loss, loss_dict = loss_fn(score, feat, target, target_cam,
                                          text_proj=text_proj, text_embs=text_emb,
                                          student_tokens=student_tokens, teacher_tokens=teacher_tokens,
                                          dist_output=dist_output, img_paths=img_paths)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)
            
            if "gram_loss" in loss_dict:
                gram_loss_meter.update(loss_dict["gram_loss"], img.shape[0])
            if "distill_matrix_loss" in loss_dict:
                distill_loss_meter.update(loss_dict["distill_matrix_loss"], img.shape[0])

            global_step += 1

            torch.cuda.synchronize()

            # ── Per-step wandb logging ────────────────────────────────
            if use_wandb and global_step % wandb_log_freq == 0:
                base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                wandb_metrics = {
                    "train/loss_total": loss.item(),
                    "train/acc_step": acc.item() if hasattr(acc, 'item') else float(acc),
                    "train/lr": base_lr,
                    "train/epoch": epoch,
                }
                # Add individual loss components (only non-zero ones)
                for k, v in loss_dict.items():
                    if v != 0.0:
                        wandb_metrics[f"train/{k}"] = v
                wandb.log(wandb_metrics, step=global_step)

            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    if (n_iter + 1) % log_period == 0:
                        base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                        extra = ''
                        if gram_loss_meter.count > 0:
                            extra += f', GramLoss: {gram_loss_meter.avg:.3f}'
                        if distill_loss_meter.count > 0:
                            extra += f', DistillLoss: {distill_loss_meter.avg:.3f}'
                        logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}{}, Acc: {:.3f}, Base Lr: {:.2e}"
                                    .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, extra, acc_meter.avg, base_lr))
            else:
                if (n_iter + 1) % log_period == 0:
                    base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                    extra = ''
                    if gram_loss_meter.count > 0:
                        extra += f', GramLoss: {gram_loss_meter.avg:.3f}'
                    if distill_loss_meter.count > 0:
                        extra += f', DistillLoss: {distill_loss_meter.avg:.3f}'
                    logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}{}, Acc: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, extra, acc_meter.avg, base_lr))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        epoch_time = end_time - start_time
        if cfg.SOLVER.WARMUP_METHOD == 'cosine':
            scheduler.step(epoch)
        else:
            scheduler.step()

        # ── Per-epoch wandb logging ───────────────────────────────────
        if use_wandb:
            base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
            wandb.log({
                "epoch/loss_avg": loss_meter.avg,
                "epoch/acc_avg": acc_meter.avg if not hasattr(acc_meter.avg, 'item') else acc_meter.avg.item(),
                "epoch/lr": base_lr,
                "epoch/time_s": epoch_time,
                "epoch/throughput_samples_s": train_loader.batch_size / time_per_batch,
                "epoch/epoch": epoch,
            }, step=global_step)

        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per epoch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch * (n_iter + 1), train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, val_batch in enumerate(val_loader):
                        if len(val_batch) == 7:
                            img, vid, camid, camids, target_view, _, mask = val_batch
                        else:
                            img, vid, camid, camids, target_view, _ = val_batch
                            mask = None
                            
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            if mask is not None:
                                mask = mask.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view, mask=mask)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()

                    # ── wandb eval logging (DDP rank 0) ───────────────
                    if use_wandb:
                        wandb.log({
                            "eval/mAP": mAP,
                            "eval/rank1": cmc[0],
                            "eval/rank5": cmc[4],
                            "eval/rank10": cmc[9],
                            "eval/epoch": epoch,
                        }, step=global_step)
            else:
                model.eval()
                for n_iter, val_batch in enumerate(val_loader):
                    if len(val_batch) == 7:
                        img, vid, camid, camids, target_view, _, mask = val_batch
                    else:
                        img, vid, camid, camids, target_view, _ = val_batch
                        mask = None
                        
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        if mask is not None:
                            mask = mask.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view, mask=mask)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()

                # ── wandb eval logging (single GPU) ───────────────
                if use_wandb:
                    wandb.log({
                        "eval/mAP": mAP,
                        "eval/rank1": cmc[0],
                        "eval/rank5": cmc[4],
                        "eval/rank10": cmc[9],
                        "eval/epoch": epoch,
                    }, step=global_step)


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, val_batch in enumerate(val_loader):
        if len(val_batch) == 7:
            img, pid, camid, camids, target_view, imgpath, mask = val_batch
        else:
            img, pid, camid, camids, target_view, imgpath = val_batch
            mask = None
            
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            if mask is not None:
                mask = mask.to(device)
            feat = model(img, cam_label=camids, view_label=target_view, mask=mask)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]



