from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import random
import torch
import logging
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []

        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)
        logger = logging.getLogger("transreid.check")
        logger.info("Dataset statistics:")
        logger.info("  ----------------------------------------")
        logger.info("  subset   | # ids | # images | # cameras")
        logger.info("  ----------------------------------------")
        logger.info("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        logger.info("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        logger.info("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        logger.info("  ----------------------------------------")

class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, guided_attention=False):
        self.dataset = dataset
        self.transform = transform
        self.guided_attention = guided_attention
        if guided_attention and len(dataset) > 0:
            first_img_path = dataset[0][0]
            first_mask_path = first_img_path.replace('bounding_box_train', 'bounding_box_train_masks_results')
            first_mask_path = first_mask_path.replace('bounding_box_test', 'bounding_box_test_masks_results')
            first_mask_path = first_mask_path.replace('query', 'query_masks_results')
            first_mask_path = first_mask_path.replace('.jpg', '.png')
            mask_dir = osp.dirname(first_mask_path)
            if osp.exists(mask_dir):
                print(f"Guided Attention: Mask directory verified at '{mask_dir}'")
            else:
                print(f"Guided Attention WARNING: Mask directory NOT found at '{mask_dir}'. Using all-zero masks.")


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)

        if self.guided_attention:
            mask_path = img_path.replace('bounding_box_train', 'bounding_box_train_masks_results')
            mask_path = mask_path.replace('bounding_box_test', 'bounding_box_test_masks_results')
            mask_path = mask_path.replace('query', 'query_masks_results')
            mask_path = mask_path.replace('.jpg', '.png')
            
            got_mask = False
            if osp.exists(mask_path):
                try:
                    mask = Image.open(mask_path).convert('L')
                    got_mask = True
                except IOError:
                    pass
            
            if not got_mask:
                mask = Image.new('L', img.size, 0)
                
            if self.transform is not None:
                if hasattr(self.transform, 'transforms'):
                    for t in self.transform.transforms:
                        if t.__class__.__name__ in ['Normalize', 'ColorJitter']:
                            img = t(img)
                        elif t.__class__.__name__ == 'RandomErasing':
                            img_before = img.clone()
                            img = t(img)
                            diff = (img_before != img).any(dim=0)
                            if isinstance(mask, torch.Tensor):
                                mask[:, diff] = 0.0
                        else:
                            state_random = random.getstate()
                            state_torch = torch.get_rng_state()
                            
                            img = t(img)
                            
                            random.setstate(state_random)
                            torch.set_rng_state(state_torch)
                            mask = t(mask)
                else:
                    state_random = random.getstate()
                    state_torch = torch.get_rng_state()
                    img = self.transform(img)
                    random.setstate(state_random)
                    torch.set_rng_state(state_torch)
                    mask = self.transform(mask)
                
                if isinstance(mask, torch.Tensor):
                    mask = (mask > 0.5).float()
            
            return img, pid, camid, trackid, img_path, mask

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, trackid, img_path


class ImageDatasetWithText(Dataset):
    """
    Wraps the raw dataset list to additionally return a
    SigLIP2 text embedding.  Returns zeros for missing keys.

    Supports two lookup strategies controlled by ``lookup_by_pid``:
      - True  → dict keyed by PID  (int):  {pid: tensor(768,)}
      - False → dict keyed by filename (str): {"0002_c1s1_000451_03.jpg": tensor(768,)}
    """
    def __init__(self, dataset, transform, text_emb_dict,
                 text_emb_dim=768, lookup_by_pid=True, guided_attention=False):
        """
        Args:
            dataset        : list of (img_path, pid, camid, trackid)
            transform      : torchvision transforms
            text_emb_dict  : dict {key: tensor(768,)}
            text_emb_dim   : dimensionality of text embeddings
            lookup_by_pid  : if True, key = pid (int); if False, key = basename(img_path)
        """
        self.dataset       = dataset
        self.transform     = transform
        self.text_emb_dict = text_emb_dict
        self.text_emb_dim  = text_emb_dim
        self.lookup_by_pid = lookup_by_pid
        self.guided_attention = guided_attention
        if guided_attention and len(dataset) > 0:
            first_img_path = dataset[0][0]
            first_mask_path = first_img_path.replace('bounding_box_train', 'bounding_box_train_masks_results')
            first_mask_path = first_mask_path.replace('bounding_box_test', 'bounding_box_test_masks_results')
            first_mask_path = first_mask_path.replace('query', 'query_masks_results')
            first_mask_path = first_mask_path.replace('.jpg', '.png')
            mask_dir = osp.dirname(first_mask_path)
            if osp.exists(mask_dir):
                print(f"Guided Attention: Mask directory verified at '{mask_dir}'")
            else:
                print(f"Guided Attention WARNING: Mask directory NOT found at '{mask_dir}'. Using all-zero masks.")


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)

        if self.guided_attention:
            mask_path = img_path.replace('bounding_box_train', 'bounding_box_train_masks_results')
            mask_path = mask_path.replace('bounding_box_test', 'bounding_box_test_masks_results')
            mask_path = mask_path.replace('query', 'query_masks_results')
            mask_path = mask_path.replace('.jpg', '.png')
            
            got_mask = False
            if osp.exists(mask_path):
                try:
                    mask = Image.open(mask_path).convert('L')
                    got_mask = True
                except IOError:
                    pass
            
            if not got_mask:
                mask = Image.new('L', img.size, 0)
                
            if self.transform is not None:
                if hasattr(self.transform, 'transforms'):
                    for t in self.transform.transforms:
                        if t.__class__.__name__ in ['Normalize', 'ColorJitter']:
                            img = t(img)
                        elif t.__class__.__name__ == 'RandomErasing':
                            img_before = img.clone()
                            img = t(img)
                            diff = (img_before != img).any(dim=0)
                            if isinstance(mask, torch.Tensor):
                                mask[:, diff] = 0.0
                        else:
                            state_random = random.getstate()
                            state_torch = torch.get_rng_state()
                            img = t(img)
                            random.setstate(state_random)
                            torch.set_rng_state(state_torch)
                            mask = t(mask)
                else:
                    state_random = random.getstate()
                    state_torch = torch.get_rng_state()
                    img = self.transform(img)
                    random.setstate(state_random)
                    torch.set_rng_state(state_torch)
                    mask = self.transform(mask)
                
                if isinstance(mask, torch.Tensor):
                    mask = (mask > 0.5).float()
        else:
            mask = None
            if self.transform is not None:
                img = self.transform(img)

        # Choose lookup key based on strategy
        if self.lookup_by_pid:
            key = pid
        else:
            key = osp.basename(img_path)

        text_emb = self.text_emb_dict.get(key, torch.zeros(self.text_emb_dim))

        if self.guided_attention:
            return img, pid, camid, trackid, img_path, text_emb, mask
        else:
            return img, pid, camid, trackid, img_path, text_emb

