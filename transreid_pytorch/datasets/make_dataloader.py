import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset, ImageDatasetWithText
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler, RandomIdentitySampler_IdUniform
from .market1501 import Market1501
from .msmt17 import MSMT17
from .dukemtmcreid import DukeMTMCreID
from .occ_duke import OCC_DukeMTMCreID
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .mm import MM
__factory = {
    'market1501': Market1501,
    'msmt17': MSMT17,
    'dukemtmc': DukeMTMCreID,
    'occ_duke': OCC_DukeMTMCreID,
    'mm': MM,
}

def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    if len(batch[0]) == 6:
        imgs, pids, camids, viewids, img_paths, masks = zip(*batch)
        pids = torch.tensor(pids, dtype=torch.int64)
        viewids = torch.tensor(viewids, dtype=torch.int64)
        camids = torch.tensor(camids, dtype=torch.int64)
        return torch.stack(imgs, dim=0), pids, camids, viewids, img_paths, torch.stack(masks, dim=0)
    else:
        imgs, pids, camids, viewids, img_paths = zip(*batch)
        pids = torch.tensor(pids, dtype=torch.int64)
        viewids = torch.tensor(viewids, dtype=torch.int64)
        camids = torch.tensor(camids, dtype=torch.int64)
        return torch.stack(imgs, dim=0), pids, camids, viewids, img_paths

def train_collate_fn_text(batch):
    """
    Collate function for training with text embeddings.
    The 5th element is img_path, 6th is text embedding, 7th is mask (if guided attention is true).
    """
    if len(batch[0]) == 7:
        imgs, pids, camids, viewids, img_paths, text_embs, masks = zip(*batch)
        pids = torch.tensor(pids, dtype=torch.int64)
        viewids = torch.tensor(viewids, dtype=torch.int64)
        camids = torch.tensor(camids, dtype=torch.int64)
        text_embs = torch.stack(text_embs, dim=0)
        return torch.stack(imgs, dim=0), pids, camids, viewids, img_paths, text_embs, torch.stack(masks, dim=0)
    else:
        imgs, pids, camids, viewids, img_paths, text_embs = zip(*batch)
        pids = torch.tensor(pids, dtype=torch.int64)
        viewids = torch.tensor(viewids, dtype=torch.int64)
        camids = torch.tensor(camids, dtype=torch.int64)
        text_embs = torch.stack(text_embs, dim=0)
        return torch.stack(imgs, dim=0), pids, camids, viewids, img_paths, text_embs

def val_collate_fn(batch):
    if len(batch[0]) == 6:
        imgs, pids, camids, viewids, img_paths, masks = zip(*batch)
        viewids = torch.tensor(viewids, dtype=torch.int64)
        camids_batch = torch.tensor(camids, dtype=torch.int64)
        return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths, torch.stack(masks, dim=0)
    else:
        imgs, pids, camids, viewids, img_paths = zip(*batch)
        viewids = torch.tensor(viewids, dtype=torch.int64)
        camids_batch = torch.tensor(camids, dtype=torch.int64)
        return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths

def make_dataloader(cfg):
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    if cfg.DATASETS.NAMES == 'ourapi':
        dataset = OURAPI(root_train=cfg.DATASETS.ROOT_TRAIN_DIR, root_val=cfg.DATASETS.ROOT_VAL_DIR, config=cfg)
    else:
        dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

    # --- Load text embeddings if configured ---
    text_emb_dict = {}
    use_text = False
    if cfg.MODEL.TEXT_EMB_PATH:
        text_emb_dict = torch.load(cfg.MODEL.TEXT_EMB_PATH, map_location="cpu", weights_only=False)
        use_text = True
        print(f"Loaded {len(text_emb_dict)} text embeddings from {cfg.MODEL.TEXT_EMB_PATH}")

    if use_text:
        train_set = ImageDatasetWithText(
            dataset.train, train_transforms,
            text_emb_dict, text_emb_dim=cfg.MODEL.TEXT_EMB_DIM,
            lookup_by_pid=cfg.MODEL.TEXT_EMB_BY_PID,
            guided_attention=cfg.MODEL.GUIDED_ATTENTION_TRAIN
        )
        collate_fn_train = train_collate_fn_text
    else:
        train_set = ImageDataset(dataset.train, train_transforms, guided_attention=cfg.MODEL.GUIDED_ATTENTION_TRAIN)
        collate_fn_train = train_collate_fn

    train_set_normal = ImageDataset(dataset.train, val_transforms, guided_attention=cfg.MODEL.GUIDED_ATTENTION_TEST)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    if cfg.DATALOADER.SAMPLER in ['softmax_triplet', 'img_triplet']:
        print('using img_triplet sampler')
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=collate_fn_train,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=collate_fn_train
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=collate_fn_train
        )
    elif cfg.DATALOADER.SAMPLER in ['id_triplet', 'id']:
        print('using ID sampler')
        train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler_IdUniform(dataset.train, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=collate_fn_train, drop_last = True,
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms, guided_attention=cfg.MODEL.GUIDED_ATTENTION_TEST)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    train_loader_normal = DataLoader(
        train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, train_loader_normal, val_loader, len(dataset.query), num_classes, cam_num, view_num
