from pytracking.tracker.base import BaseTracker
import ltr.data.bounding_box_utils as bbutils

import torch
import torch.nn.functional as F
import math
import time

import sys

class PySOT(BaseTracker):
    def initialize(self, image, info: dict):
        sys.path.append(self.params.path_to_pysot_install)

        from pysot.core.config import cfg
        from pysot.models.model_builder import ModelBuilder
        from pysot.tracker.tracker_builder import build_tracker

        # load config
        cfg.merge_from_file(self.params.config_file)
        cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
        device = torch.device('cuda' if cfg.CUDA else 'cpu')

        # create model
        model = ModelBuilder()

        # load model
        model.load_state_dict(torch.load(self.params.model_file,
                                         map_location=lambda storage, loc: storage.cpu()))
        model.eval().to(device)

        # build tracker
        self.tracker = build_tracker(model)
        self.tracker.init(image, info["init_bbox"])

    def track(self, image, info: dict = None):

        output_state = self.tracker.track(image)

        out = {'target_bbox': output_state['bbox']}
        return out