from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
import h5py

from util import log

__PATH__ = './datasets/shapenet'
ang_interval = 10
ang_skip = 2
rs = np.random.RandomState(123)


class Dataset(object):

    def __init__(self, ids, n, object_class, name='default',
                 max_examples=None, is_train=True):
        self._ids = list(ids)
        self.name = name
        self.is_train = is_train
        self.n = n
        self.bound = int(360/ang_interval+1)

        if max_examples is not None:
            self._ids = self._ids[:max_examples]

        filename = 'data_{}.hdf5'.format(object_class)

        file = osp.join(__PATH__, filename)
        log.info("Reading %s ...", file)

        self.data = h5py.File(file, 'r')
        log.info("Reading Done: %s", file)

    def get_data(self, id):
        if isinstance(id, bytes):
            id = id.decode("utf-8")
        # preprocessing and data augmentation
        image = 1 - self.data[id]['image'].value/255.*2
        pose = np.expand_dims(self.data[id]['pose'].value, -1)
        idx = np.concatenate(
            (np.linspace(-self.bound, 0, self.bound+1)[:-1],
             np.linspace(0, self.bound, self.bound+1)[1:])
        ) * ang_skip
        np.random.shuffle(idx)
        idx = idx[:self.n]
        ang = (idx + pose[0]).astype(np.int32)
        for a in ang:
            id_base = id.split('_')[0]
            h = id.split('_')[-1]
            id_target = '_'.join([id_base, str(a % 36), str(h)])
            image_tmp = 1 - self.data[id_target]['image'].value/255.*2
            pose_tmp = np.expand_dims(self.data[id_target]['pose'].value, -1)
            image = np.concatenate((image, image_tmp), axis=-1)
            pose = np.concatenate((pose, pose_tmp), axis=-1)
        # pose to one hot vector
        pose_one_hot = np.zeros((int(360/ang_interval) + 3, pose.shape[-1]))
        for i in range(pose.shape[-1]):
            pose_one_hot[pose[0, i], i] = 1
            pose_one_hot[int(360/ang_interval)+int(pose[1, i]/10), i] = 1
        return image, pose_one_hot

    def get_data_by_id(self, id_list):
        if isinstance(id_list[0], bytes):
            id_list = [id.decode("utf-8") for id in id_list]
        # preprocessing and data augmentation
        # taget idx: [diff ang, diff evelation]
        id = id_list[0]
        image = 1 - self.data[id]['image'].value/255.*2
        pose = np.expand_dims(self.data[id]['pose'].value, -1)

        for id_source in id_list[1:]:
            if not pose.shape[-1] > self.n:
                image_tmp = 1 - self.data[id_source]['image'].value/255.*2
                pose_tmp = np.expand_dims(self.data[id_source]['pose'].value, -1)
                image = np.concatenate((image, image_tmp), axis=-1)
                pose = np.concatenate((pose, pose_tmp), axis=-1)
        # pose to one hot vector
        pose_one_hot = np.zeros((int(360/ang_interval) + 3, pose.shape[-1]))
        for i in range(pose.shape[-1]):
            pose_one_hot[pose[0, i], i] = 1
            pose_one_hot[int(360/ang_interval)+int(pose[1, i]/10), i] = 1
        return image, pose_one_hot

    @property
    def ids(self):
        return self._ids

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        return 'Dataset (%s, %d examples)' % (
            self.name,
            len(self)
        )


def create_default_splits(n, object_class, is_train=True):
    ids_train, ids_test = all_ids(object_class)

    dataset_train = Dataset(ids_train, n, object_class, 
                            name='train', is_train=is_train)
    dataset_test = Dataset(ids_test, n, object_class,
                           name='test', is_train=is_train)
    return dataset_train, dataset_test


def all_ids(object_class):

    with open(osp.join(__PATH__, 'id_{}_train.txt'.format(object_class)), 'r') as fp:
        ids_train = [s.strip() for s in fp.readlines() if s]
    rs.shuffle(ids_train)

    with open(osp.join(__PATH__, 'id_{}_test.txt'.format(object_class)), 'r') as fp:
        ids_test = [s.strip() for s in fp.readlines() if s]
    rs.shuffle(ids_test)

    return ids_train, ids_test
