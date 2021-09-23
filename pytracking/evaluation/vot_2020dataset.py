import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList


class VOT2020Dataset(BaseDataset):
    """
    VOT2020 dataset

    Publication:
        The sixth Visual Object Tracking VOT2018 challenge results.
        Matej Kristan, Ales Leonardis, Jiri Matas, Michael Felsberg, Roman Pfugfelder, Luka Cehovin Zajc, Tomas Vojir,
        Goutam Bhat, Alan Lukezic et al.
        ECCV, 2018
        https://prints.vicos.si/publications/365

    Download the dataset from http://www.votchallenge.net/vot2018/dataset.html
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.vot_2020_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        sequence_path = sequence_name
        nz = 8
        ext = 'jpg'
        start_frame = 1

        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)
        gt = open(anno_path, 'r')
        masks = gt.readlines()
        ground_truth_rect = np.zeros((len(masks), 4))
        for index, line in enumerate(masks):
            bbox_coord = line.split(',')
            ground_truth_rect[index, 0] = int(bbox_coord[0].replace("m",""))
            ground_truth_rect[index, 1] = int(bbox_coord[1])
            ground_truth_rect[index, 2] = int(bbox_coord[2])
            ground_truth_rect[index, 3] = int(bbox_coord[3])

        end_frame = ground_truth_rect.shape[0]

        frames = ['{base_path}/{sequence_path}/color/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
                  sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext)
                  for frame_num in range(start_frame, end_frame+1)]

        # Convert gt
        if ground_truth_rect.shape[1] > 4:
            gt_x_all = ground_truth_rect[:, [0, 2, 4, 6]]
            gt_y_all = ground_truth_rect[:, [1, 3, 5, 7]]

            x1 = np.amin(gt_x_all, 1).reshape(-1,1)
            y1 = np.amin(gt_y_all, 1).reshape(-1,1)
            x2 = np.amax(gt_x_all, 1).reshape(-1,1)
            y2 = np.amax(gt_y_all, 1).reshape(-1,1)

            ground_truth_rect = np.concatenate((x1, y1, x2-x1, y2-y1), 1)
        return Sequence(sequence_name, frames, 'vot2020', ground_truth_rect)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list= ['agility',
                        'ants1',
                        'ball2',
                        'ball3',
                        'basketball',
                        'birds1',
                        'bolt1',
                        'book',
                        'butterfly',
                        'car1',
                        'conduction1',
                        'crabs1',
                        'dinosaur',
                        'dribble',
                        'drone1',
                        'drone_across',
                        'drone_flip',
                        'fernando',
                        'fish1',
                        'fish2',
                        'flamingo1',
                        'frisbee',
                        'girl',
                        'glove',
                        'godfather',
                        'graduate',
                        'gymnastics1',
                        'gymnastics2',
                        'gymnastics3',
                        'hand',
                        'hand02',
                        'hand2',
                        'handball1',
                        'handball2',
                        'helicopter',
                        'iceskater1',
                        'iceskater2',
                        'lamb',
                        'leaves',
                        'marathon',
                        'matrix',
                        'monkey',
                        'motocross1',
                        'nature',
                        'polo',
                        'rabbit',
                        'rabbit2',
                        'road',
                        'rowing',
                        'shaking',
                        'singer2',
                        'singer3',
                        'soccer1',
                        'soccer2',
                        'soldier',
                        'surfing',
                        'tiger',
                        'wheel',
                        'wiper',
                        'zebrafish1']

        return sequence_list
