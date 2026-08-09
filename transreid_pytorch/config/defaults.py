from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
_C.MODEL.NAME = 'resnet50'
# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = ''
_C.MODEL.PRETRAIN_HW_RATIO = 1

# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' , 'self' , 'finetune'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'

# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IF_WITH_CENTER = 'no'

_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0

_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# If train with multi-gpu ddp mode, options: 'True', 'False'
_C.MODEL.DIST_TRAIN = False
# If train with soft triplet loss, options: 'True', 'False'
_C.MODEL.NO_MARGIN = False
# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'
# If train with arcface loss, options: 'True', 'False'
_C.MODEL.COS_LAYER = False

_C.MODEL.DROPOUT_RATE = 0.0
# Reduce feature dim
_C.MODEL.REDUCE_FEAT_DIM = False
_C.MODEL.FEAT_DIM = 512
# Transformer setting
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0
_C.MODEL.TRANSFORMER_TYPE = 'None'
_C.MODEL.STRIDE_SIZE = [16, 16]
_C.MODEL.POOLING = 'None'   # gem, avg, max, avg_max
_C.MODEL.STEM_CONV = False

# ── Guided Attention ──────────────────────────────────────────────
_C.MODEL.GUIDED_ATTENTION_TRAIN = False   # enable mask-guided attention during training
_C.MODEL.GUIDED_ATTENTION_TEST  = False   # enable mask-guided attention during inference
_C.MODEL.MASK_THRESHOLD         = 0.3    # patch overlap threshold to classify as "human"
_C.MODEL.GUIDED_SCALE_RATIO     = 1.1    # additive boost ratio for CLS→human-patch attention

# ── KoLeo Loss ────────────────────────────────────────────────────
_C.MODEL.USE_KOLEO_LOSS    = False
_C.MODEL.KOLEO_LOSS_WEIGHT = 0.1

# ── Supervised Contrastive Loss ───────────────────────────────────
_C.MODEL.USE_SUPCON_LOSS    = False
_C.MODEL.SUPCON_TEMPERATURE = 0.05
_C.MODEL.SUPCON_LOSS_WEIGHT = 1.0

# ── Gram Anchor Loss ──────────────────────────────────────────────
_C.MODEL.USE_GRAM_ANCHOR_LOSS = False
_C.MODEL.GRAM_ANCHOR_TEACHER_TYPE = 'dinov3'  # 'dinov3' or 'personvit'
_C.MODEL.GRAM_ANCHOR_TEACHER_MODEL_KEY = 'personvit-vit_small'
_C.MODEL.GRAM_ANCHOR_TEACHER_CHECKPOINT = ''
_C.MODEL.GRAM_ANCHOR_DINO_MODEL = 'dinov3-vits16-pretrain'
_C.MODEL.GRAM_ANCHOR_STUDENT_DIM = 384
_C.MODEL.GRAM_ANCHOR_TEACHER_DIM = 384
_C.MODEL.GRAM_ANCHOR_LOSS_WEIGHT = 1.0
_C.MODEL.HF_TOKEN = ''

# JPM Parameter
_C.MODEL.JPM = False
_C.MODEL.SHIFT_NUM = 5
_C.MODEL.SHUFFLE_GROUP = 2
_C.MODEL.DEVIDE_LENGTH = 4
_C.MODEL.RE_ARRANGE = True

# SIE Parameter
_C.MODEL.SIE_COE = 3.0
_C.MODEL.SIE_CAMERA = False
_C.MODEL.SIE_VIEW = False

# ── Text Alignment (VLM) ──────────────────────────────────────────
_C.MODEL.TEXT_EMB_PATH       = "scripts/qwen_8B_v1.pt"        # path to .pt file with text embeddings
_C.MODEL.TEXT_EMB_BY_PID     = True       # True = dict keyed by PID (int); False = keyed by filename (str)
_C.MODEL.TEXT_EMB_DIM        = 768       # SigLIP2 output dimension
_C.MODEL.TEXT_ALIGN_WEIGHT   = 0.3       # λ weighting text loss (projector mode)
_C.MODEL.TEXT_ALIGN_LOSS     = "infonce" # "infonce", "text_center", or "cosine" (projector mode)
_C.MODEL.TEXT_ALIGN_TEMPERATURE = 0.07   # InfoNCE softmax temperature (projector mode)
_C.MODEL.TEXT_PROJ_HEAD      = False     # True = projector MLP + warmup; False = dist_loss family
_C.MODEL.TEXT_PROJ_WARMUP_EPOCHS = 5     # freeze backbone for first N epochs (projector mode only)

# ── Dist-loss settings (active when TEXT_PROJ_HEAD = False) ────────
_C.MODEL.TEXT_LOSS_TYPE  = "kl"          # "l1", "mse", "huber", or "kl"
_C.MODEL.TEXT_LOSS_HYPR  = 0.07          # tau (KL), delta (Huber); ignored for L1/MSE
_C.MODEL.TEXT_LOSS_WEIGHT = 0.5          # λ multiplier for dist_loss

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [384, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [384, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('market1501')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('../data')
_C.DATASETS.ROOT_TRAIN_DIR = ('../data')
_C.DATASETS.ROOT_VAL_DIR = ('../data')


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16
# remove tail data
_C.DATALOADER.REMOVE_TAIL = 0

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 100
# Base learning rate
_C.SOLVER.BASE_LR = 3e-4
# Whether using larger learning rate for fc layer
_C.SOLVER.LARGE_FC_LR = False
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 1
# Factor of learning bias
_C.SOLVER.SEED = 1234
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005

# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005

# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (40, 70)
# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.01
#  warm up epochs
_C.SOLVER.WARMUP_EPOCHS = 5
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "cosine"

_C.SOLVER.COSINE_MARGIN = 0.5
_C.SOLVER.COSINE_SCALE = 30

# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 10
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 10
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 128, each GPU will
# contain 16 images per batch
_C.SOLVER.IMS_PER_BATCH = 64
_C.SOLVER.TRP_L2 = False

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #

_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 128
# If test with re-ranking, options: 'True','False'
_C.TEST.RE_RANKING = False
# Path to trained model
_C.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'after'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'

# Name for saving the distmat after testing.
_C.TEST.DIST_MAT = "dist_mat.npy"
# Whether calculate the eval score option: 'True', 'False'
_C.TEST.EVAL = False
# ---------------------------------------------------------------------------- #
# Weights & Biases (wandb)
# ---------------------------------------------------------------------------- #
_C.WANDB = CN()
_C.WANDB.ENABLE = False                # master switch for wandb logging
_C.WANDB.PROJECT = "person-reid"       # wandb project name
_C.WANDB.RUN_NAME = ""                 # wandb run name (empty = auto-generated)
_C.WANDB.ENTITY = ""                   # wandb team / user (empty = default)
_C.WANDB.TAGS = ()                     # tuple of string tags
_C.WANDB.NOTES = ""                    # free-text run description
_C.WANDB.LOG_FREQ = 10                 # log metrics every N training steps
_C.WANDB.LOG_DISTILL_METRICS = False   # log distillation-specific metrics

# ---------------------------------------------------------------------------- #
# Distillation (DeiT-style dual-token)
# ---------------------------------------------------------------------------- #
_C.DISTILL = CN()
_C.DISTILL.ENABLED = False                   # master switch — false = zero change to existing behaviour
_C.DISTILL.TEACHER_CHECKPOINT_PATH = ''      # path to frozen teacher checkpoint (B/16)
_C.DISTILL.TEACHER_EMBEDDINGS_PATH = ''      # path to precomputed teacher embeddings .pt
_C.DISTILL.TEACHER_CACHE_VIEW = 'deterministic'  # 'deterministic' or 'augmented'
_C.DISTILL.LAMBDA = 1.0                      # static multiplier for distillation loss
_C.DISTILL.MATRIX_LOSS_TYPE = 'l1'           # 'l1', 'l2', 'huber'
_C.DISTILL.MATRIX_LOSS_HYPR = 0.3            # delta for huber; ignored for l1/l2
_C.DISTILL.USE_DIST_TOKEN_AT_EVAL = True     # avg(CLS,DIST) at eval; forced false when ENABLED=false
_C.DISTILL.DIST_TOKEN_INIT = 'copy_cls'      # initialisation strategy

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""
