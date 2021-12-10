import os
import cv2
import random
import numpy as np
import lmdb
import logging
import six
from tqdm import tqdm
from PIL import Image
from PIL import ImageFile
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

ImageFile.LOAD_TRUNCATED_IMAGES = True

from transformer_ocr.utils.image_processing import resize_img, get_new_width


class OCRDataset(Dataset):
    """Dumps or loads dataset for OCR task. This followed by:
    https://github.com/clovaai/deep-text-recognition-benchmark

    Args:
        saved_path (str):
        root_dir (str):
        gt_path (str):
        vocab_builder ():
        img_height (int):
        img_width_min (int):
        img_width_max (int):
        transform ():
        max_readers (int):
    """

    def __init__(self, saved_path,
                 root_dir,
                 gt_path,
                 vocab_builder,
                 img_height,
                 img_width_min,
                 img_width_max,
                 transform,
                 max_readers):

        # Image info
        self.img_height = img_height
        self.img_width_min = img_width_min
        self.img_width_max = img_width_max

        # In/out info
        self.root_dir = root_dir
        self.gt_path = os.path.join(root_dir, gt_path)
        self.saved_path = saved_path

        # Vocabulary builder to encode text
        self.vocab_builder = vocab_builder

        # Image transformation if any
        self.transform = transform

        # Validate whether ground-truth path
        if not os.path.exists(self.gt_path):
            logging.error('{} not exists. Please verify this!'.format(self.gt_path))
            exit(0)

        if not os.path.exists(self.saved_path):
            self.write_data_to_disk()
        else:
            logging.info('{} exists!'.format(self.saved_path))

        # Read LMDB file from the saved path
        self.env = lmdb.open(
            self.saved_path,
            max_readers=max_readers,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)
        self.txn = self.env.begin(write=False)

        dataset_size = int(self.txn.get('num-samples'.encode()))
        self.dataset_size = dataset_size

        self.clusters = self.build_clusters_by_img_width()

    def build_clusters_by_img_width(self) -> defaultdict:
        clusters = defaultdict(list)

        dataset_indices = tqdm(range(self.dataset_size),
                               desc='{} build clusters'.format(self.saved_path),
                               ncols=100,
                               leave=True,
                               position=0)

        def get_kernel(data_idx):
            key = 'dim-%09d' % data_idx
            spatial_info = self.txn.get(key.encode())
            img_h, img_w = np.fromstring(spatial_info, dtype=np.int32)

            new_w = get_new_width(img_w, img_h, self.img_height, self.img_width_min, self.img_width_max)

            return new_w

        for idx in dataset_indices:
            clusters[get_kernel(idx)].append(idx)

        return clusters

    def load_cache(self, idx):
        """Load data from cache."""
        binary_img = 'image-%09d' % idx
        binary_img = self.txn.get(binary_img.encode())

        label = 'label-%09d' % idx
        label = self.txn.get(label.encode()).decode()

        img_path = 'path-%09d' % idx
        img_path = self.txn.get(img_path.encode()).decode()

        img_buf = six.BytesIO()
        img_buf.write(binary_img)
        img_buf.seek(0)

        return img_buf, label, img_path

    def read_data(self, idx):
        img_buf, label, img_path = self.load_cache(idx)

        sentence = self.vocab_builder.encode(label)

        img = Image.open(img_buf).convert('RGB')
        if self.transform:
            img = self.transform(img)

        resized_img = resize_img(img, self.img_height, self.img_width_min, self.img_width_max)

        return resized_img, sentence, img_path

    def write_data_to_disk(self):
        """Create LMDB dataset to save on disk."""
        with open(self.gt_path, mode='r', encoding='utf-8-sig') as f:
            lines = f.readlines()
            annotations = [line.strip().split('\t') for line in lines]

        dataset_size = len(annotations)

        # Create Environment for LMDB
        env = lmdb.open(self.saved_path, map_size=1e12)
        cache = {}
        cnt = 0
        error = 0

        pbar = tqdm(range(dataset_size), ncols=100, desc='Generate {} ...'.format(self.saved_path))
        for i in pbar:
            if len(annotations[i]) > 2:
                file_name, *labels = annotations[i]
                label = " ".join([*labels])
                logging.warning("This {} might contains more than 2 elements!".format(file_name))
            elif len(annotations[i]) < 2:
                logging.warning("Ignore this!")
                continue
            else:
                file_name, label = annotations[i]

            img_path = os.path.join(self.root_dir, file_name)
            if not os.path.exists(img_path):
                error += 1
                logging.error('This image path: {} doest not exist!'.format(img_path))
                continue

            with open(img_path, 'rb') as f:
                binary_img = f.read()

            is_valid, img_h, img_w = self.check_image_is_valid(binary_img)

            if not is_valid:
                error += 1
                continue

            imageKey = 'image-%09d' % cnt
            labelKey = 'label-%09d' % cnt
            pathKey = 'path-%09d' % cnt
            dimKey = 'dim-%09d' % cnt

            cache[imageKey] = binary_img
            cache[labelKey] = label.encode()
            cache[pathKey] = file_name.encode()
            cache[dimKey] = np.array([img_h, img_w], dtype=np.int32).tobytes()

            cnt += 1

            if cnt % 1000 == 0:
                self.write_cache(env, cache)
                cache = {}

        dataset_size = cnt - 1
        cache['num-samples'] = str(dataset_size).encode()
        self.write_cache(env, cache)

        if error > 0:
            logging.warning('Ignore {} invalid images'.format(error))

        logging.info('Created {} samples in the dataset'.format(dataset_size))
        env.close()

    def __getitem__(self, idx):
        img, sentence, img_path = self.read_data(idx)

        img_path = os.path.join(self.root_dir, img_path)

        sample = {'img': img, 'sentence': sentence, 'img_path': img_path}

        return sample

    def __len__(self):
        return self.dataset_size

    @staticmethod
    def check_image_is_valid(binary_img):
        is_valid = True
        img_h = None
        img_w = None

        image_buf = np.fromstring(binary_img, dtype=np.uint8)
        try:
            img = cv2.imdecode(image_buf, cv2.IMREAD_GRAYSCALE)
            img_h, img_w = img.shape[0], img.shape[1]
            if img_h * img_w == 0:
                is_valid = False
        except Exception:
            is_valid = False
            logging.error('Error occurred when reading binary image!')

        return is_valid, img_h, img_w

    @staticmethod
    def write_cache(env, cache):
        with env.begin(write=True) as txn:
            for k, v in cache.items():
                txn.put(k.encode(), v)


class ClusterRandomSampler(Sampler):
    """Takes a dataset with cluster_indices property, cuts it into batch-sized chunks
    Drops the extra items, not fitting into exact batches

    Args:
        data_source (Dataset): a Dataset to sample from. Should have a cluster_indices property
        batch_size (int): a batch size that you would like to use later with Dataloader class
        shuffle (bool): whether to shuffle the data or not

    This is implemented by repo: https://github.com/snakers4/mnasnet-pytorch
    """
    def __init__(self, data_source, batch_size, shuffle=True):
        super(ClusterRandomSampler, self).__init__()

        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle

        batch_lists = []
        for cluster, cluster_indices in self.data_source.clusters.items():
            if self.shuffle:
                random.shuffle(cluster_indices)

            batches = [cluster_indices[i:i + self.batch_size] for i in range(0, len(cluster_indices), self.batch_size)]
            batches = [_ for _ in batches if len(_) == self.batch_size]
            if self.shuffle:
                random.shuffle(batches)

            batch_lists.append(batches)

        lst = self.flatten_list(batch_lists)
        if self.shuffle:
            random.shuffle(lst)

        self.lst = lst

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.lst)

        return iter(self.flatten_list(self.lst))

    def __len__(self):
        return len(self.data_source)

    @staticmethod
    def flatten_list(lst):
        return [item for sublist in lst for item in sublist]


class Collator(object):
    def __call__(self, batch):
        img_path = []
        img = []
        tgt_input = []
        max_label_len = max(len(sample['sentence']) for sample in batch)

        for sample in batch:
            img.append(sample['img'])
            img_path.append(sample['img_path'])
            sentence = sample['sentence']

            sentence_len = len(sentence)
            tgt = np.concatenate((
                sentence,
                np.zeros(max_label_len - sentence_len, dtype=np.int32)))
            tgt_input.append(tgt)

        img = np.array(img, dtype=np.float32)
        tgt_input = np.array(tgt_input, dtype=np.int64).T

        rs = {
            'img': torch.FloatTensor(img),
            'tgt_output': torch.LongTensor(tgt_input.T),
            'img_path': img_path
        }

        return rs

