from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer, WarmupMultiStepLR
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_train
import random
import torch
import numpy as np
import os
import argparse
from config import cfg
import torch.distributed as dist

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    cfg.freeze()
    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    try:
        os.makedirs(output_dir)
    except:
        pass

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info(args)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            #  logger.info(config_str)

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    logger.info("Running with config:\n{}".format(cfg))

    # ── Weights & Biases ──────────────────────────────────────────────
    wandb_run = None
    if cfg.WANDB.ENABLE:
        import wandb
        wandb_run = wandb.init(
            project=cfg.WANDB.PROJECT,
            name=cfg.WANDB.RUN_NAME or None,
            entity=cfg.WANDB.ENTITY or None,
            tags=list(cfg.WANDB.TAGS) if cfg.WANDB.TAGS else None,
            notes=cfg.WANDB.NOTES or None,
            config=dict(cfg),          # log the full flat config
            dir=output_dir,
            reinit=True,
        )
        logger.info("wandb initialized – project: %s, run: %s",
                     cfg.WANDB.PROJECT, wandb_run.name)
        # Log distillation hyperparameters once for run-comparison clarity
        if cfg.DISTILL.ENABLED:
            wandb.config.update({
                'distill_lambda': cfg.DISTILL.LAMBDA,
                'distill_matrix_loss_type': cfg.DISTILL.MATRIX_LOSS_TYPE,
                'distill_teacher_cache_view': cfg.DISTILL.TEACHER_CACHE_VIEW,
                'distill_dist_token_init': cfg.DISTILL.DIST_TOKEN_INIT,
            }, allow_val_change=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)

    # Let wandb track gradients & parameters
    if wandb_run is not None:
        import wandb
        wandb.watch(model, log="all", log_freq=cfg.WANDB.LOG_FREQ)

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    if cfg.SOLVER.WARMUP_METHOD == 'cosine':
        logger.info('===========using cosine learning rate=======')
        scheduler = create_scheduler(cfg, optimizer)
    else:
        logger.info('===========using normal learning rate=======')
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                      cfg.SOLVER.WARMUP_FACTOR,
                                      cfg.SOLVER.WARMUP_EPOCHS, cfg.SOLVER.WARMUP_METHOD)

    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_query, args.local_rank
    )

    # ── Cleanup wandb ─────────────────────────────────────────────────
    if wandb_run is not None:
        import wandb
        wandb.finish()

