import torch
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms as T
import glob
import re
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class Hema((ImageDataset)):
    def __init__(self, root='/home/zbc/data/reid', test_set_dir='',
                    train_dir='', query_dir='', gallery_dir='',
                    verbose=True, **kwargs):
        self.dataset_dir = root
        self.test_set_dir = test_set_dir
        self.train_dir = osp.join(self.dataset_dir, train_dir)
        self.query_dir = osp.join(self.dataset_dir, query_dir)
        self.gallery_dir = osp.join(self.dataset_dir, gallery_dir)

        self._check_before_run()
        if osp.isdir(train_dir):
            train = self._process_dir(self.train_dir, relabel=True)
        else:
            train = self._from_list(self.train_dir, self.dataset_dir, relabel=True)

        if osp.isdir(query_dir):
            query = self._process_dir(self.query_dir, relabel=False)
        else:
            query = self._from_list(self.query_dir, self.test_set_dir, relabel=False)

        if osp.isdir(gallery_dir):
            gallery = self._process_dir(self.gallery_dir, relabel=False)
        else:
            gallery = self._from_list(self.gallery_dir, self.test_set_dir, relabel=False)

        # if verbose:
        #     print("=> Data loaded")
        #     self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs,num_cams

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c([\d]+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))
        return dataset


    def _from_list(self, fn, root, relabel=False, is_train=True):
        pid_container = {}
        dataset = []
        with open(fn, "r") as f:
            lines = f.readlines()
            for l in lines:
                l = l.strip()
                if len(l.split(' ')) == 2:
                    car_name, car_class, car_camid = l.split(' ')[0], int(l.split(' ')[1]), -1
                elif len(l.split(' ')) == 1 and not is_train:
                    car_name, car_class, car_camid = l.split(' ')[0], -1, -1
                elif len(l.split(' ')) == 3:
                    car_name, car_class, car_camid = l.split(' ')[0], int(l.split(' ')[1]), int(l.split(' ')[2])
                else:
                    print("dataset wrong")

                if car_class not in pid_container:
                    pid_container[car_class] = len(pid_container)
                if relabel:
                    dataset.append((osp.join(root,car_name), int(pid_container[car_class]), int(car_camid)))
                else:
                    dataset.append((osp.join(root, car_name), int(car_class), int(car_camid)))

        return dataset
