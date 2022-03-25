"""
TODO: Clean up code, right now prefix: xs = extremum summary functions, mh = multi-hypothesis functions
"""
import random

from pytracking.tracker.base import BaseTracker
import torch
import torch.nn.functional as F
import math
import time
from pytracking import dcf, TensorList
from pytracking.features.preprocessing import numpy_to_torch
from pytracking.utils.plotting import show_tensor, plot_graph
from pytracking.features.preprocessing import sample_patch_multiscale, sample_patch_transformed, sample_patch
from pytracking.features import augmentation
import ltr.data.bounding_box_utils as bbutils
from ltr.models.target_classifier.initializer import FilterInitializerZero
from ltr.models.layers import activation

from pytracking.analysis.extract_results import calc_err_center, calc_iou_overlap

import cv2 as cv

#additional mode for k centers summary
import pytracking.libs.kcenters as kc
import pytracking.libs.visualization as vs
import matplotlib.pyplot as plt
import copy

class TrainingSample(object):
    def __init__(self):
        self.sample = None
        self.target_box = None
        self.im_patch = None
        self.weight = 0
        self.id = None
        self.num_supported_hypotheses = 0

class MH_Node(object):
    def __init__(self):
        self.net = None
        self.target_filter = None

        self.curr_feats = None
        self.curr_scores = None
        self.mem_ids = set()
        self.frame_ids = set()
        self.scores = []

        # extremum_summary related
        self.extremum_threshold = 0

    def clone(self):
        #todo: just deepcopy the whole object
        new_hypo = MH_Node()
        new_hypo.net = copy.deepcopy(self.net)
        new_hypo.target_filter = copy.deepcopy(self.target_filter)
        new_hypo.frame_ids = copy.deepcopy(self.frame_ids)
        new_hypo.mem_ids = copy.deepcopy(self.mem_ids)
        new_hypo.scores = copy.deepcopy(self.scores)
        new_hypo.extremum_threshold = self.extremum_threshold
        return new_hypo

class MH_DiMP(BaseTracker):

    multiobj_mode = 'parallel'

    def _read_image(self, image_file: str):
        im = cv.imread(image_file)
        return cv.cvtColor(im, cv.COLOR_BGR2RGB)

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.params.net.initialize()
        self.features_initialized = True

    def initialize(self, image, info: dict) -> dict:
        # Initialize some stuff
        self.frame_num = 1
        if not self.params.has('device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'

        # Initialize network
        self.initialize_features()

        # The DiMP network
        self.net = self.params.net

        # Time initialization
        tic = time.time()

        # Convert image
        im = numpy_to_torch(image)

        # Initialize dashboard image holder
        self.summary_patches = torch.zeros(self.params.get('summary_size', 15), self.params.image_sample_size, self.params.image_sample_size, 3)
        self.ref_patches = torch.zeros(self.params.get('sample_memory_size'), self.params.image_sample_size, self.params.image_sample_size, 3)

        self.summary_update = False
        self.summary_prev_ind = -1
        self.summary_stored_samples = 0

        self.del_summary_num = -1
        self.summary_size = [self.params.get('summary_size', 15)]
        self.lock_mode = False
        self.in_default = False

        # store computed thresholds
        self.use_saved_threshold = False
        self.saved_threshold = 0

        # Get target position and size
        state = info['init_bbox']
        self.pos = torch.Tensor([state[1] + (state[3] - 1)/2, state[0] + (state[2] - 1)/2])
        self.target_sz = torch.Tensor([state[3], state[2]])

        # Get object id
        self.object_id = info.get('object_ids', [None])[0]
        self.id_str = '' if self.object_id is None else ' {}'.format(self.object_id)

        # Set sizes
        self.image_sz = torch.Tensor([im.shape[2], im.shape[3]])
        sz = self.params.image_sample_size
        sz = torch.Tensor([sz, sz] if isinstance(sz, int) else sz)
        if self.params.get('use_image_aspect_ratio', False):
            sz = self.image_sz * sz.prod().sqrt() / self.image_sz.prod().sqrt()
            stride = self.params.get('feature_stride', 32)
            sz = torch.round(sz / stride) * stride
        self.img_sample_sz = sz
        self.img_support_sz = self.img_sample_sz

        # Set search area
        search_area = torch.prod(self.target_sz * self.params.search_area_scale).item()
        self.target_scale =  math.sqrt(search_area) / self.img_sample_sz.prod().sqrt()

        # Target size in base scale
        self.base_target_sz = self.target_sz / self.target_scale

        # Setup scale factors
        if not self.params.has('scale_factors'):
            self.params.scale_factors = torch.ones(1)
        elif isinstance(self.params.scale_factors, (list, tuple)):
            self.params.scale_factors = torch.Tensor(self.params.scale_factors)

        # Setup scale bounds
        self.min_scale_factor = torch.max(10 / self.base_target_sz)
        self.max_scale_factor = torch.min(self.image_sz / self.base_target_sz)

        # Extract and transform sample
        init_backbone_feat = self.generate_init_samples(im)

        # Initialize classifier
        self.init_classifier(init_backbone_feat)

        # Initialize IoUNet
        if self.params.get('use_iou_net', True):
            self.init_iou_net(init_backbone_feat)

        # If used, initialize multi-hypothesis trackers
        if self.params.get("use_mh", False):
            new_hypothesis = MH_Node()
            new_hypothesis.target_filter = self.target_filter
            new_hypothesis.net = self.net
            new_hypothesis.mem_ids = set(range(self.num_init_samples[0]))
            new_hypothesis.extremum_threshold = kc.threshold_cost(self.training_samples[0][:self.num_init_samples[0],...])

            self.hypotheses = []
            self.hypotheses.append(new_hypothesis)

            self.mh_training_sample_set = {}

        # If used, initialize online active summaries
        if self.params.get("use_active_online_summary", False):
            assert(self.summary_size[0] > 0)

            self.initial_net = copy.deepcopy(self.net)
            self.initial_target_filter = self.target_filter

            # Start with the first image for extremum summaries
            self.training_samples[0][self.num_init_samples[0], ...] = self.training_samples[0][0, ...]
            self.target_boxes[self.num_init_samples[0],...] = self.target_boxes[0, ...]
            self.summary_patches[0,...] = self.ref_patches[0,...]
            self.sample_weights[0][self.num_init_samples[0]] = self.sample_weights[0][0]
            self.num_stored_samples[0] += 1

            self.summary_updated = False
            self.query_requested = False

            if self.params.get("use_active_online_extremum", False):
                self.extremum_summary_threshold, _ = kc.get_mean_summary_score(self.training_samples[0][:self.num_init_samples[0],...],
                                                                    dist_func=self.params.get("dist_func", "cosine_dist"))

        # If used, initialize global trained networks (uses ground truth information)
        if self.params.get("use_global_trainer", False):

            # Extremum summary sample selection (greedy and global)
            if self.params.get("use_global_extremum", True):
                print("Setting up global extremum samples...")
                summary_samples = torch.zeros((self.summary_size[0], *self.training_samples[0].shape[1:]), device=self.params.device)
                target_boxes = torch.zeros((self.summary_size[0],4), device=self.params.device)
                summary_patches = self.summary_patches.new_zeros(self.summary_patches.size())
                summary_weights = torch.zeros(self.summary_size[0], device=self.params.device)

                # start by adding in the first sample
                summary_samples[0,...] = self.training_samples[0][0]
                target_boxes[0,...] = self.target_boxes[0]
                summary_patches[0] = self.ref_patches[0]
                summary_weights[0] = self.sample_weights[0][0]

                curr_summary_size = 1
                if self.params.get("dist_func", "cosine_dist") is "cosine_dist":
                    dist_func = kc.cosine_dist
                elif self.params.get("dist_func") is "l2_normalised_dist":
                    dist_func = kc.l2_normalised_dist
                else:
                    dist_func = kc.l2_dist

                threshold = kc.threshold_cost(self.training_samples[0][:self.num_init_samples[0]], distance_function=dist_func)
                thresholds = []
                thresholds.append(threshold)

                sample_inds = [0] * self.summary_size[0]

                for i, im_info in enumerate(zip(info["all_frames"], info["all_bboxes"])):
                    im_path, bbox = im_info
                    image = self._read_image(im_path)
                    im = numpy_to_torch(image)

                    sample, target_box, im_patches = self.get_sample_and_box(im, bbox)
                    replace_ind, _ = kc.online_summary_update_index_extremum(summary_samples[:curr_summary_size,...],
                                                                             sample, self.summary_size[0],
                                                                             threshold=threshold,
                                                                             distance_function=dist_func)

                    if replace_ind > -1:
                        summary_samples[replace_ind,...] = sample
                        target_boxes[replace_ind,...] = target_box
                        img = im_patches.reshape(im_patches.size()[1:])
                        img = torch.moveaxis((img - img.min()) / (img.max() - img.min()), 0, 2)
                        summary_patches[replace_ind,...] = img
                        summary_weights[replace_ind] = self.sample_weights[0][0] # todo: make this a parameter

                        if curr_summary_size < self.summary_size[0]:
                            curr_summary_size += 1

                        if self.params.get("use_mean_score", True):
                            threshold = kc.threshold_cost(summary_samples[:curr_summary_size,...], distance_function=dist_func)
                        else:
                            threshold = self.params.get("summary_threshold_gamma", 1.005) * threshold
                        thresholds.append(threshold)

                        sample_inds[replace_ind] = i

                print("Used inds: ", sample_inds)
                print("Training...")
                print(thresholds)
                self.training_samples[0] = torch.cat((self.training_samples[0][:self.num_init_samples[0],...], summary_samples[:curr_summary_size]), 0)
                self.target_boxes = torch.cat((self.target_boxes[:self.num_init_samples[0],...], target_boxes[:curr_summary_size]), 0)
                self.summary_patches = summary_patches[:curr_summary_size]
                self.sample_weights[0] = torch.cat((self.sample_weights[0][:self.num_init_samples[0],...], summary_weights[:curr_summary_size]), 0)

            # Uniform random sampling approach
            else:

                total_frames = len(info["all_frames"])
                sample_inds = random.sample(range(total_frames), self.params.get("sample_memory_size",50)-self.num_init_samples[0])
                sample_inds.sort()

                print("Used frames: ", sample_inds)

                # Construct training sample set
                for i, sample_ind in enumerate(sample_inds):
                    im_path = info["all_frames"][sample_ind]
                    state = info["all_bboxes"][sample_ind]

                    image = self._read_image(im_path)
                    im = numpy_to_torch(image)

                    sample, target_box, im_patches = self.get_sample_and_box(im, state)

                    self.training_samples[0][i+self.num_init_samples[0], ...] = sample
                    self.target_boxes[i+self.num_init_samples[0], ...] = target_box
                    img = im_patches.reshape(im_patches.size()[1:])
                    img = torch.moveaxis((img - img.min()) / (img.max() - img.min()), 0, 2)
                    self.summary_patches[i, ...] = img
                    self.ref_patches[i+self.num_init_samples[0],...] = img
                    self.sample_weights[0][i+self.num_init_samples[0]] = self.sample_weights[0][0] #todo: implement this correctly

            num_iter = 2  # todo: make this smarter?
            plot_loss = self.params.debug > 0

            # Run the filter optimizer module
            with torch.no_grad():
                self.target_filter, _, losses = self.net.classifier.filter_optimizer(
                    self.target_filter,
                    num_iter=num_iter,
                    feat=self.training_samples[0],
                    bb=self.target_boxes,
                    sample_weight=self.sample_weights[0],
                    compute_losses=plot_loss)
        if self.params.get("log_summary", False):
            out = {'time': time.time() - tic,
                   'summary_threshold': self.extremum_summary_threshold.item(),
                   'query_requested': self.query_requested,
                   'summary_size': self.num_stored_samples[0]-self.num_init_samples[0]}
        else:
            out = {'time': time.time() - tic}
        return out

    def get_sample_and_box(self, im, bbox):
        # Convert image to latent sample and gt bbox into the correct sampling coords

        pos = torch.Tensor([bbox[1] + (bbox[3] - 1) / 2, bbox[0] + (bbox[2] - 1) / 2])
        target_sz = torch.Tensor([bbox[3], bbox[2]])

        image_sz = torch.Tensor([im.shape[2], im.shape[3]])
        sz = self.params.image_sample_size
        sz = torch.Tensor([sz, sz] if isinstance(sz, int) else sz)
        if self.params.get('use_image_aspect_ratio', False):
            sz = image_sz * sz.prod().sqrt() / image_sz.prod().sqrt()
            stride = self.params.get('feature_stride', 32)
            sz = torch.round(sz / stride) * stride
        img_sample_sz = sz

        search_area = torch.prod(target_sz * self.params.search_area_scale).item()
        target_scale = math.sqrt(search_area) / img_sample_sz.prod().sqrt()

        # Compute augmentation size
        aug_expansion_factor = self.params.get('augmentation_expansion_factor', None)
        aug_expansion_sz = self.img_sample_sz.clone()
        aug_output_sz = None
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.img_sample_sz * aug_expansion_factor).long()
            aug_expansion_sz += (aug_expansion_sz - self.img_sample_sz.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = img_sample_sz.long().tolist()

        #todo: this should simply use extract backbone, but it's not working for some reason
        global_shift = torch.zeros(2)
        transforms = [augmentation.Identity(aug_output_sz, global_shift.long().tolist())]
        im_patches = sample_patch_transformed(im, pos, target_scale, aug_expansion_sz, transforms)

        #im_patches = sample_patch_transformed(im, pos, target_scale * self.params.scale_factors, img_sample_sz, transforms)

        # backbone_feat, sample_coords, im_patches = self.extract_backbone_features(im, pos,
        #                                                                           target_scale * self.params.scale_factors,
        #                                                                           img_sample_sz)
        with torch.no_grad():
            backbone_feat = self.net.extract_backbone(im_patches)

        sample = self.get_classification_features(backbone_feat)

        sample_scale = target_scale
        sample_pos = pos.round()
        target_box = self.get_iounet_box(pos, target_sz, sample_pos, sample_scale)

        return sample, target_box, im_patches

    def track(self, image, info: dict = None) -> dict:
        self.debug_info = {}

        self.frame_num += 1
        self.debug_info['frame_num'] = self.frame_num

        if self.params.get("use_oracle_feedback", False):
            self.ground_truth_bbox = info['gt']

        # Convert image
        im = numpy_to_torch(image)

        if self.params.get("use_mh") and \
                self.params.prune_random_sample and \
                len(self.mh_training_sample_set) > self.params.summary_size:
            self.mh_prune_random_frame_id()

        # Print useful multi-hypothesis tracking information
        if self.params.get("use_mh", True):
            print("Num hypo: ", len(self.hypotheses))
            print("Num samples: ", len(self.mh_training_sample_set))
            print([x.frame_ids for x in self.hypotheses])
            print([(x, self.mh_training_sample_set[x].num_supported_hypotheses) for x in self.mh_training_sample_set])

        # ------- LOCALIZATION ------- #

        # Extract backbone features
        backbone_feat, sample_coords, im_patches = self.extract_backbone_features(im, self.get_centered_sample_pos(),
                                                                      self.target_scale * self.params.scale_factors,
                                                                      self.img_sample_sz)
        # Extract classification features
        if not self.params.get("use_mh", True):
            test_x = self.get_classification_features(backbone_feat)
        else:
            self.mh_get_classification_features(backbone_feat)

        # Location of sample
        sample_pos, sample_scales = self.get_sample_location(sample_coords)

        # Compute classification scores
        if not self.params.get("use_mh", True):
            scores_raw = self.classify_target(test_x)
        else:
            scores_raw, test_x = self.mh_classify_target()

        # Localize the target
        translation_vec, scale_ind, s, flag = self.localize_target(scores_raw, sample_pos, sample_scales)
        new_pos = sample_pos[scale_ind,:] + translation_vec

        # Update position and scale
        if flag != 'not_found':
            if self.params.get('use_iou_net', True):
                update_scale_flag = self.params.get('update_scale_when_uncertain', True) or flag != 'uncertain'
                if self.params.get('use_classifier', True):
                    self.update_state(new_pos)
                self.refine_target_box(backbone_feat, sample_pos[scale_ind,:], sample_scales[scale_ind], scale_ind, update_scale_flag)
            elif self.params.get('use_classifier', True):
                self.update_state(new_pos, sample_scales[scale_ind])

        # ------- UPDATE ------- #

        update_flag = flag not in ['not_found', 'uncertain']
        hard_negative = (flag == 'hard_negative')
        learning_rate = self.params.get('hard_negative_learning_rate', None) if hard_negative else None

        if update_flag and self.params.get('update_classifier', False):
            # Get train sample
            train_x = test_x[scale_ind:scale_ind+1, ...]

            # Create target_box and label for spatial sample
            target_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos[scale_ind,:], sample_scales[scale_ind])

            # Update the classifier model
            if not self.params.get('skip_update_classifier', False):
                self.update_classifier(train_x, target_box, im_patches, learning_rate, s[scale_ind,...])

        # Set the pos of the tracker to iounet pos
        if self.params.get('use_iou_net', True) and flag != 'not_found' and hasattr(self, 'pos_iounet'):
            self.pos = self.pos_iounet.clone()

        score_map = s[scale_ind, ...]
        max_score = torch.max(score_map).item()

        # Visualize and set debug info
        self.search_area_box = torch.cat((sample_coords[scale_ind,[1,0]], sample_coords[scale_ind,[3,2]] - sample_coords[scale_ind,[1,0]] - 1))
        self.debug_info['flag' + self.id_str] = flag
        self.debug_info['max_score' + self.id_str] = max_score
        if self.visdom is not None:
            self.visdom.register(score_map, 'heatmap', 2, 'Score Map' + self.id_str)
            self.visdom.register(self.debug_info, 'info_dict', 1, 'Status')
        elif self.params.debug >= 2:
            show_tensor(score_map, 5, title='Max score = {:.2f}'.format(max_score))

        # Compute output bounding box
        new_state = torch.cat((self.pos[[1,0]] - (self.target_sz[[1,0]]-1)/2, self.target_sz[[1,0]]))

        if self.params.get('output_not_found_box', False) and flag == 'not_found':
            output_state = [-1, -1, -1, -1]
        else:
            output_state = new_state.tolist()

        # todo: we can fake delays by looking at the whole gt history
        if self.params.get('use_mh') and self.params.get('use_oracle_feedback', False):
            oracle_iou_threshold = self.params.get('oracle_iou_threshold', 0.85)
            oracle_prec_threshold = self.params.get('oracle_prec_threshold', 5)
            pred_bb = torch.tensor(output_state)
            gt = info["gt"]
            iou = calc_iou_overlap(torch.tensor(gt).unsqueeze(0), pred_bb.unsqueeze(0))

            if self.params.get('use_oracle_iou') and self.frame_num in self.mh_training_sample_set and iou < oracle_iou_threshold:
                self.mh_prune_frame_id(self.frame_num)

        if self.summary_update:
            self.summary_update = False
            index_to_replace = self.summary_prev_ind
            out = {'target_bbox': output_state, 'index_to_replace': index_to_replace}
        else:
            out = {'target_bbox': output_state}

        if self.params.get("log_summary", False):
            out['summary_threshold'] = self.extremum_summary_threshold.item()
            out['query_requested'] = self.query_requested
            out['summary_size'] = self.num_stored_samples[0] - self.num_init_samples[0]
        return out

    def get_sample_location(self, sample_coord):
        """Get the location of the extracted sample."""
        sample_coord = sample_coord.float()
        sample_pos = 0.5*(sample_coord[:,:2] + sample_coord[:,2:] - 1)
        sample_scales = ((sample_coord[:,2:] - sample_coord[:,:2]) / self.img_sample_sz).prod(dim=1).sqrt()
        return sample_pos, sample_scales

    def get_centered_sample_pos(self):
        """Get the center position for the new sample. Make sure the target is correctly centered."""
        return self.pos + ((self.feature_sz + self.kernel_size) % 2) * self.target_scale * \
               self.img_support_sz / (2*self.feature_sz)

    def classify_target(self, sample_x: TensorList):
        """Classify target by applying the DiMP filter."""
        with torch.no_grad():
            scores = self.net.classifier.classify(self.target_filter, sample_x)
        return scores

    def mh_classify_target(self):
        # TODO: This is incredibly naive
        best_max_score = -float("Inf")
        best_score = None
        best_feats = None

        # Get classification scores for all hypotheses, return the best one as current estimate
        for i, hypothesis in enumerate(self.hypotheses):
            with torch.no_grad():
                hypothesis.curr_scores = hypothesis.net.classifier.classify(hypothesis.target_filter,
                                                                            hypothesis.curr_feats)
                max_score = torch.max(hypothesis.curr_scores)
                hypothesis.scores.append(max_score)

                if max_score > best_max_score:
                    best_max_score = max_score
                    best_score = hypothesis.curr_scores
                    best_feats = hypothesis.curr_feats
        return best_score, best_feats

    def localize_target(self, scores, sample_pos, sample_scales):
        """Run the target localization."""

        scores = scores.squeeze(1)

        preprocess_method = self.params.get('score_preprocess', 'none')
        if preprocess_method == 'none':
            pass
        elif preprocess_method == 'exp':
            scores = scores.exp()
        elif preprocess_method == 'softmax':
            reg_val = getattr(self.net.classifier.filter_optimizer, 'softmax_reg', None)
            scores_view = scores.view(scores.shape[0], -1)
            scores_softmax = activation.softmax_reg(scores_view, dim=-1, reg=reg_val)
            scores = scores_softmax.view(scores.shape)
        else:
            raise Exception('Unknown score_preprocess in params.')

        score_filter_ksz = self.params.get('score_filter_ksz', 1)
        if score_filter_ksz > 1:
            assert score_filter_ksz % 2 == 1
            kernel = scores.new_ones(1,1,score_filter_ksz,score_filter_ksz)
            scores = F.conv2d(scores.view(-1,1,*scores.shape[-2:]), kernel, padding=score_filter_ksz//2).view(scores.shape)

        if self.params.get('advanced_localization', False):
            return self.localize_advanced(scores, sample_pos, sample_scales)

        # Get maximum
        score_sz = torch.Tensor(list(scores.shape[-2:]))
        score_center = (score_sz - 1)/2
        max_score, max_disp = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score, dim=0)
        max_disp = max_disp[scale_ind,...].float().cpu().view(-1)
        target_disp = max_disp - score_center

        # Compute translation vector and scale change factor
        output_sz = score_sz - (self.kernel_size + 1) % 2
        translation_vec = target_disp * (self.img_support_sz / output_sz) * sample_scales[scale_ind]

        return translation_vec, scale_ind, scores, None

    def localize_advanced(self, scores, sample_pos, sample_scales):
        """Run the target advanced localization (as in ATOM)."""

        sz = scores.shape[-2:]
        score_sz = torch.Tensor(list(sz))
        output_sz = score_sz - (self.kernel_size + 1) % 2
        score_center = (score_sz - 1)/2

        scores_hn = scores
        if self.output_window is not None and self.params.get('perform_hn_without_windowing', False):
            scores_hn = scores.clone()
            scores *= self.output_window

        max_score1, max_disp1 = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score1, dim=0)
        sample_scale = sample_scales[scale_ind]
        max_score1 = max_score1[scale_ind]
        max_disp1 = max_disp1[scale_ind,...].float().cpu().view(-1)
        target_disp1 = max_disp1 - score_center
        translation_vec1 = target_disp1 * (self.img_support_sz / output_sz) * sample_scale
        if max_score1.item() < self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores_hn, 'not_found'
        if max_score1.item() < self.params.get('uncertain_threshold', -float('inf')):
            return translation_vec1, scale_ind, scores_hn, 'uncertain'
        if max_score1.item() < self.params.get('hard_sample_threshold', -float('inf')):
            return translation_vec1, scale_ind, scores_hn, 'hard_negative'

        # Mask out target neighborhood
        target_neigh_sz = self.params.target_neighborhood_scale * (self.target_sz / sample_scale) * (output_sz / self.img_support_sz)

        tneigh_top = max(round(max_disp1[0].item() - target_neigh_sz[0].item() / 2), 0)
        tneigh_bottom = min(round(max_disp1[0].item() + target_neigh_sz[0].item() / 2 + 1), sz[0])
        tneigh_left = max(round(max_disp1[1].item() - target_neigh_sz[1].item() / 2), 0)
        tneigh_right = min(round(max_disp1[1].item() + target_neigh_sz[1].item() / 2 + 1), sz[1])
        scores_masked = scores_hn[scale_ind:scale_ind + 1, ...].clone()
        scores_masked[...,tneigh_top:tneigh_bottom,tneigh_left:tneigh_right] = 0

        # Find new maximum
        max_score2, max_disp2 = dcf.max2d(scores_masked)
        max_disp2 = max_disp2.float().cpu().view(-1)
        target_disp2 = max_disp2 - score_center
        translation_vec2 = target_disp2 * (self.img_support_sz / output_sz) * sample_scale

        prev_target_vec = (self.pos - sample_pos[scale_ind,:]) / ((self.img_support_sz / output_sz) * sample_scale)

        # Handle the different cases
        if max_score2 > self.params.distractor_threshold * max_score1:
            disp_norm1 = torch.sqrt(torch.sum((target_disp1-prev_target_vec)**2))
            disp_norm2 = torch.sqrt(torch.sum((target_disp2-prev_target_vec)**2))
            disp_threshold = self.params.dispalcement_scale * math.sqrt(sz[0] * sz[1]) / 2

            if disp_norm2 > disp_threshold and disp_norm1 < disp_threshold:
                return translation_vec1, scale_ind, scores_hn, 'hard_negative'
            if disp_norm2 < disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec2, scale_ind, scores_hn, 'hard_negative'
            if disp_norm2 > disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec1, scale_ind, scores_hn, 'uncertain'

            # If also the distractor is close, return with highest score
            return translation_vec1, scale_ind, scores_hn, 'uncertain'

        if max_score2 > self.params.hard_negative_threshold * max_score1 and max_score2 > self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores_hn, 'hard_negative'

        return translation_vec1, scale_ind, scores_hn, 'normal'

    def extract_backbone_features(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor):
        im_patches, patch_coords = sample_patch_multiscale(im, pos, scales, sz,
                                                           mode=self.params.get('border_mode', 'replicate'),
                                                           max_scale_change=self.params.get('patch_max_scale_change', None))
        with torch.no_grad():
            backbone_feat = self.net.extract_backbone(im_patches)
        return backbone_feat, patch_coords, im_patches

    def get_classification_features(self, backbone_feat):
        with torch.no_grad():
            return self.net.extract_classification_feat(backbone_feat)

    def mh_get_classification_features(self, backbone_feat):
        for hypothesis in self.hypotheses:
            hypothesis.curr_feats = hypothesis.net.extract_classification_feat(backbone_feat)

    def get_iou_backbone_features(self, backbone_feat):
        return self.net.get_backbone_bbreg_feat(backbone_feat)

    def get_iou_features(self, backbone_feat):
        with torch.no_grad():
            return self.net.bb_regressor.get_iou_feat(self.get_iou_backbone_features(backbone_feat))

    def get_iou_modulation(self, iou_backbone_feat, target_boxes):
        with torch.no_grad():
            return self.net.bb_regressor.get_modulation(iou_backbone_feat, target_boxes)

    def generate_init_samples(self, im: torch.Tensor) -> TensorList:
        """Perform data augmentation to generate initial training samples."""

        mode = self.params.get('border_mode', 'replicate')
        if mode == 'inside':
            # Get new sample size if forced inside the image
            im_sz = torch.Tensor([im.shape[2], im.shape[3]])
            sample_sz = self.target_scale * self.img_sample_sz
            shrink_factor = (sample_sz.float() / im_sz)
            if mode == 'inside':
                shrink_factor = shrink_factor.max()
            elif mode == 'inside_major':
                shrink_factor = shrink_factor.min()
            shrink_factor.clamp_(min=1, max=self.params.get('patch_max_scale_change', None))
            sample_sz = (sample_sz.float() / shrink_factor)
            self.init_sample_scale = (sample_sz / self.img_sample_sz).prod().sqrt()
            tl = self.pos - (sample_sz - 1) / 2
            br = self.pos + sample_sz / 2 + 1
            global_shift = - ((-tl).clamp(0) - (br - im_sz).clamp(0)) / self.init_sample_scale
        else:
            self.init_sample_scale = self.target_scale
            global_shift = torch.zeros(2)

        self.init_sample_pos = self.pos.round()

        # Compute augmentation size
        aug_expansion_factor = self.params.get('augmentation_expansion_factor', None)
        aug_expansion_sz = self.img_sample_sz.clone()
        aug_output_sz = None
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.img_sample_sz * aug_expansion_factor).long()
            aug_expansion_sz += (aug_expansion_sz - self.img_sample_sz.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = self.img_sample_sz.long().tolist()

        # Random shift for each sample
        get_rand_shift = lambda: None
        random_shift_factor = self.params.get('random_shift_factor', 0)
        if random_shift_factor > 0:
            get_rand_shift = lambda: ((torch.rand(2) - 0.5) * self.img_sample_sz * random_shift_factor + global_shift).long().tolist()

        # Always put identity transformation first, since it is the unaugmented sample that is always used
        self.transforms = [augmentation.Identity(aug_output_sz, global_shift.long().tolist())]

        augs = self.params.augmentation if self.params.get('use_augmentation', True) else {}

        # Add all augmentations
        if 'shift' in augs:
            self.transforms.extend([augmentation.Translation(shift, aug_output_sz, global_shift.long().tolist()) for shift in augs['shift']])
        if 'relativeshift' in augs:
            get_absolute = lambda shift: (torch.Tensor(shift) * self.img_sample_sz/2).long().tolist()
            self.transforms.extend([augmentation.Translation(get_absolute(shift), aug_output_sz, global_shift.long().tolist()) for shift in augs['relativeshift']])
        if 'fliplr' in augs and augs['fliplr']:
            self.transforms.append(augmentation.FlipHorizontal(aug_output_sz, get_rand_shift()))
        if 'blur' in augs:
            self.transforms.extend([augmentation.Blur(sigma, aug_output_sz, get_rand_shift()) for sigma in augs['blur']])
        if 'scale' in augs:
            self.transforms.extend([augmentation.Scale(scale_factor, aug_output_sz, get_rand_shift()) for scale_factor in augs['scale']])
        if 'rotate' in augs:
            self.transforms.extend([augmentation.Rotate(angle, aug_output_sz, get_rand_shift()) for angle in augs['rotate']])

        # Extract augmented image patches
        im_patches = sample_patch_transformed(im, self.init_sample_pos, self.init_sample_scale, aug_expansion_sz, self.transforms)
        for i, img in enumerate(im_patches):
            im_patch = torch.moveaxis((img - img.min()) / (img.max() - img.min()), 0, 2)
            self.ref_patches[i,...] = im_patch

        #store the initial image training_patches

        #UPDATE_HERE

        #for the case where the summary size is 0 we don't make patches yet
        '''

        for index, img in enumerate(im_patches):
            self.summary_patches[index] = (torch.moveaxis((img-img.min())/(img.max()-img.min()), 0, 2))

        if (im_patches.size()[0] < self.summary_size[0]):
            img = im_patches[0]
            img = (torch.moveaxis((img-img.min())/(img.max()-img.min()), 0, 2))
            for index in list(range(im_patches.size()[0], self.summary_size[0])):
                self.summary_patches[index] = img '''

        # Extract initial backbone features
        with torch.no_grad():
            init_backbone_feat = self.net.extract_backbone(im_patches)

        return init_backbone_feat

    def init_target_boxes(self):
        """Get the target bounding boxes for the initial augmented samples."""
        self.num_init_samples = [len(self.transforms)]
        self.sample_memory_size = self.num_init_samples[0] + self.summary_size[0]

        self.summary_rel_weight = self.params.get('summary_rel_weight', .5)
        self.online_start_ind = [self.summary_size[0] + self.num_init_samples[0]]
        self.classifier_target_box = self.get_iounet_box(self.pos, self.target_sz, self.init_sample_pos, self.init_sample_scale)
        init_target_boxes = TensorList()
        for T in self.transforms:
            init_target_boxes.append(self.classifier_target_box + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
        init_target_boxes = torch.cat(init_target_boxes.view(1, 4), 0).to(self.params.device)
        self.target_boxes = init_target_boxes.new_zeros(self.sample_memory_size, 4)
        self.target_boxes[:init_target_boxes.shape[0],:] = init_target_boxes
        self.mh_initial_target_boxes = init_target_boxes

        #leave this empty initially

        '''
        for i in list(range(self.num_init_samples[0],self.online_start_ind[0])):
            #initialize all of the summary_set to train_x[0]

            #UPDATE_HERE
            if (i-self.num_init_samples[0] > self.num_init_samples[0]):
                self.target_boxes[i] = self.target_boxes[0]
            else:
                self.target_boxes[i] = self.target_boxes[i - self.num_init_samples[0]]
        '''

        return init_target_boxes

    def init_memory(self, train_x: TensorList):
        # Initialize first-frame spatial training samples
        self.num_init_samples = train_x.size(0)

        # make init_sample_weights the initial augmentation sample weights
        # todo: what should the initial weights be
        # todo: what should the sample weights be (idea: probably the same in the multi-hypothesis case actually)
        init_sample_weights = TensorList([x.new_ones(1) / (x.shape[0] + self.summary_rel_weight * self.summary_size[0]) for x in train_x])
        summary_sample_weights = TensorList([self.summary_rel_weight * init_sample_weights])

        # Used for the global MH tracker
        # todo: likely don't need these/these weights should probably be the same
        self.mh_initial_sample_weight = init_sample_weights
        self.mh_online_sample_weight = summary_sample_weights[0]

        #init_sample_weights = TensorList([x.new_ones(1) / x.shape[0] for x in train_x])

        # Sample counters and weights for spatial
        self.num_stored_samples = self.num_init_samples.copy()
        #self.num_init_samples.copy()
        self.previous_replace_ind = [None] * len(self.num_stored_samples)
        self.sample_weights = TensorList([x.new_zeros(self.sample_memory_size) for x in train_x])

        # UPDATE HERE
        for i in list(range(self.online_start_ind[0])):
            if i < self.num_init_samples[0]:
                self.sample_weights[0][i] = init_sample_weights[0][0]
            else:
                self.sample_weights[0][i] = summary_sample_weights[0][0]
        '''
        for sw, init_sw, num in zip(self.sample_weights, summary_sample_weights, self.online_start_ind):
            sw[:num] = init_sw

        for sw, init_sw, num in zip(self.sample_weights, init_sample_weights, self.num_init_samples):
            sw[:num] = init_sw
        '''

        # Initialize memory
        self.sample_memory_size = self.num_init_samples[0] + self.summary_size[0]
        self.training_samples = TensorList(
            [x.new_zeros(self.sample_memory_size, x.shape[1], x.shape[2], x.shape[3]) for x in train_x])

        # todo: clean this up
        self.mh_initial_training_samples = train_x
        self.mh_initial_sample_weights = self.sample_weights[0][:self.num_init_samples[0]]

        for ts, x in zip(self.training_samples, train_x):
            ts[:x.shape[0],...] = x

        #don't make it all the first image!!!

        # don't do this for empty summary set
        '''
        for i in list(range(self.num_init_samples[0],self.online_start_ind[0])):
            #initialize all of the summary_set to train_x[0]

            #UPDATE_HERE
            #note that this requires that self.num_init_samples is the same size as the summary set
            if (i-self.num_init_samples[0] >= self.num_init_samples[0]):
                self.training_samples[0][i] = train_x[0][0]
            else:
                self.training_samples[0][i] = train_x[0][i-self.num_init_samples[0]]
                '''

    def mh_update_memory_extremum(self, sample_x: TensorList, target_box, im_patch, learning_rate = None):

        # Prune the worst-confidence until we have max number of hypotheses remaining
        max_num_hypotheses = self.params.get("max_num_hypotheses", 8)
        scores = torch.tensor([h.scores[-1] for h in self.hypotheses])
        if len(self.hypotheses) > max_num_hypotheses:
            top_k_scores, top_k_inds = torch.topk(scores, max_num_hypotheses)
            new_hypotheses = [0]*max_num_hypotheses

            # Decrement counter for training samples
            # todo: this is inefficient
            for i, h in enumerate(self.hypotheses):
                if i not in top_k_inds:
                    for frame_id in h.frame_ids:
                        self.mh_training_sample_set[frame_id].num_supported_hypotheses -= 1

            for i, k in enumerate(top_k_inds):
                new_hypotheses[i] = self.hypotheses[k]
            self.hypotheses = new_hypotheses

        # Create training sample
        new_training_sample = TrainingSample()
        new_training_sample.sample = sample_x
        new_training_sample.weight = self.mh_online_sample_weight
        new_training_sample.id = self.frame_num
        new_training_sample.target_box = target_box.to(self.params.device)
        img = im_patch.reshape(im_patch.size()[1:])
        img = torch.moveaxis((img - img.min()) / (img.max() - img.min()), 0, 2)
        new_training_sample.im_patch = img

        new_frame_id = self.frame_num
        #self.mh_training_sample_set[self.frame_num] = ts

        # Update hypotheses
        new_hypotheses = []
        for hypothesis in self.hypotheses:
            new_hypotheses.append(hypothesis)

            # todo: big question, how do we initialize these? Should we be allowed to replace the initial sample?
            # guessing no, maybe the first sample can be included and we always start with that as the seed
            # Check to see if we should add sample to hypothesis
            # for now, we ignore the initial set, we'll come back to this later
            frame_ids = list(hypothesis.frame_ids)
            num_online_samples = len(frame_ids)
            samples = torch.zeros(torch.Size([num_online_samples]) + self.mh_initial_training_samples[0].shape[1:], device=self.params.device)
            for i, frame_id in enumerate(frame_ids):
                samples[i, ...] = self.mh_training_sample_set[frame_id].sample[0]

            max_summary_size = self.params.get('summary_size', 35)
            # todo: what should be going into this?
            if hypothesis.extremum_threshold is 0:
                replace_ind, threshold = kc.online_summary_update_index_extremum(samples, new_training_sample.sample[0], max_summary_size)
            else:
                replace_ind, threshold = kc.online_summary_update_index_extremum(samples, new_training_sample.sample[0], max_summary_size, threshold=hypothesis.extremum_threshold)

            # Sample not added, no need to add a new hypothesis
            if replace_ind < 0:
                continue

            # Add new hypothesis with the new sample
            new_hypothesis = hypothesis.clone()
            new_hypothesis.frame_ids.add(new_frame_id)

            if threshold is not None:
                new_hypothesis.extremum_threshold = threshold * self.params.get('summary_threshold_gamma', 2)

            if new_frame_id not in self.mh_training_sample_set:
                self.mh_training_sample_set[new_frame_id] = new_training_sample

            # If was added (and not replaced)
            if replace_ind > len(frame_ids) - 1:
                samples = torch.cat((samples, new_training_sample.sample[0]), dim=0)
                frame_ids.append(new_frame_id)
            # If replaced a sample
            else:
                old_frame_id = frame_ids[replace_ind]
                frame_ids[replace_ind] = new_frame_id
                new_hypothesis.frame_ids.remove(old_frame_id)
                samples[replace_ind, ...] = new_training_sample.sample[0]

            for frame_id in new_hypothesis.frame_ids:
                self.mh_training_sample_set[frame_id].num_supported_hypotheses += 1

            num_online_samples = len(new_hypothesis.frame_ids)
            num_init_samples = self.num_init_samples[0]
            num_training_samples = num_init_samples + num_online_samples
            samples = torch.cat((self.mh_initial_training_samples[0], samples), dim=0)
            sample_weights = torch.zeros([num_training_samples], device=self.params.device)
            target_boxes = torch.zeros([num_training_samples, 4], device=self.params.device)

            # todo: vectorize this!?
            # todo: look at the losses, do these updates matter?
            sample_weights[:num_init_samples, ...] = self.mh_initial_sample_weights
            target_boxes[:num_init_samples, ...] = self.mh_initial_target_boxes
            for i, frame_id in enumerate(frame_ids):
                samples[num_init_samples + i, ...] = self.mh_training_sample_set[frame_id].sample[0]
                sample_weights[num_init_samples + i, ...] = self.mh_training_sample_set[frame_id].weight[0]
                target_boxes[num_init_samples + i, ...] = self.mh_training_sample_set[frame_id].target_box

            num_iter = 2  # todo: make this smarter?
            plot_loss = self.params.debug > 0

            # Run the filter optimizer module
            with torch.no_grad():
                new_hypothesis.target_filter, _, losses = new_hypothesis.net.classifier.filter_optimizer(
                    new_hypothesis.target_filter,
                    num_iter=num_iter,
                    feat=samples,
                    bb=target_boxes,
                    sample_weight=sample_weights,
                    compute_losses=plot_loss)
            # todo: this seems dumb

            new_hypotheses.append(new_hypothesis)
        self.hypotheses = new_hypotheses

        self.mh_prune_training_sample_memory()

    def mh_update_memory(self, sample_x: TensorList, target_box, im_patch, learning_rate = None):

        # Prune the worst-confidence until we have max number of hypotheses remaining
        max_num_hypotheses = self.params.get("max_num_hypotheses", 8)
        scores = torch.tensor([h.scores[-1] for h in self.hypotheses])
        if len(self.hypotheses) > max_num_hypotheses:
            top_k_scores, top_k_inds = torch.topk(scores, max_num_hypotheses)
            new_hypotheses = [0]*max_num_hypotheses

            # Decrement counter for training samples
            # todo: this is inefficient
            for i, h in enumerate(self.hypotheses):
                if i not in top_k_inds:
                    for frame_id in h.frame_ids:
                        self.mh_training_sample_set[frame_id].num_supported_hypotheses -= 1

            for i, k in enumerate(top_k_inds):
                new_hypotheses[i] = self.hypotheses[k]
            self.hypotheses = new_hypotheses

        # Add sample to the training set
        ts = TrainingSample()
        ts.sample = sample_x
        ts.weight = self.mh_online_sample_weight
        ts.id = self.frame_num
        ts.target_box = target_box.to(self.params.device)
        img = im_patch.reshape(im_patch.size()[1:])
        img = torch.moveaxis((img - img.min()) / (img.max() - img.min()), 0, 2)
        ts.im_patch = img
        self.mh_training_sample_set[self.frame_num] = ts

        # Update hypotheses
        new_hypotheses = []
        for hypothesis in self.hypotheses:
            new_hypothesis = hypothesis.clone()
            new_hypothesis.frame_ids.add(self.frame_num)

            for frame_id in new_hypothesis.frame_ids:
                self.mh_training_sample_set[frame_id].num_supported_hypotheses += 1

            # new_hypothesis.mem_ids.add(summary_replace_ind)
            num_online_samples = len(new_hypothesis.frame_ids)
            num_init_samples = self.num_init_samples[0]
            num_training_samples = num_init_samples + num_online_samples
            samples = torch.zeros(torch.Size([num_training_samples]) + self.mh_initial_training_samples[0].shape[1:], device=self.params.device)
            sample_weights = torch.zeros([num_training_samples], device=self.params.device)
            target_boxes = torch.zeros([num_training_samples, 4], device=self.params.device)

            # todo: vectorize this!?
            # todo: look at the losses, do these updates matter?
            frame_ids = new_hypothesis.frame_ids
            samples[:num_init_samples, ...] = self.mh_initial_training_samples[0]
            sample_weights[:num_init_samples, ...] = self.mh_initial_sample_weights
            target_boxes[:num_init_samples, ...] = self.mh_initial_target_boxes
            for i, frame_id in enumerate(frame_ids):
                samples[num_init_samples + i, ...] = self.mh_training_sample_set[frame_id].sample[0]
                sample_weights[num_init_samples + i, ...] = self.mh_training_sample_set[frame_id].weight[0]
                target_boxes[num_init_samples + i, ...] = self.mh_training_sample_set[frame_id].target_box

            num_iter = 2  # todo: make this smarter?
            plot_loss = self.params.debug > 0

            # Run the filter optimizer module
            with torch.no_grad():
                new_hypothesis.target_filter, _, losses = new_hypothesis.net.classifier.filter_optimizer(
                    new_hypothesis.target_filter,
                    num_iter=num_iter,
                    feat=samples,
                    bb=target_boxes,
                    sample_weight=sample_weights,
                    compute_losses=plot_loss)
            # todo: this seems dumb
            new_hypotheses.append(hypothesis)
            new_hypotheses.append(new_hypothesis)
        self.hypotheses = new_hypotheses

        self.mh_prune_training_sample_memory()

    def mh_prune_training_sample_memory(self):
        # Prune global training sample memory
        global_frame_ids = list(self.mh_training_sample_set.keys())
        for frame_id in global_frame_ids:
            if self.mh_training_sample_set[frame_id].num_supported_hypotheses <= 0:
                self.mh_training_sample_set.pop(frame_id, None)

    def mh_prune_frame_id(self, pruned_frame_id):
        if pruned_frame_id not in self.mh_training_sample_set:
            return

        kept_hypotheses = []
        forget_inds = []
        for i, hypothesis in enumerate(self.hypotheses):
            if pruned_frame_id in hypothesis.frame_ids:
                forget_inds.append(i)
                continue
            kept_hypotheses.append(hypothesis)

        # todo: this is a common degenerate case, what is the best way to handle this?
        # possibilities: (1) ignore this id, (2) remove form all the hypotheses, but don't prune the hypotheses (future hypothese will hopefully get slightly more corrected)
        # ideal case: use this as negative training example instead
        if len(kept_hypotheses) is 0:
            print(f"Partial pruning of {pruned_frame_id} due to degenerate case")
            for i in forget_inds:
                h = self.hypotheses[i]
                h.frame_ids.remove(pruned_frame_id)

            self.mh_training_sample_set.pop(pruned_frame_id, None)
            return

        # If we do prune, update global training memory counters
        for i in forget_inds:
            h = self.hypotheses[i]
            for frame_id in h.frame_ids:
                self.mh_training_sample_set[frame_id].num_supported_hypotheses -= 1

        self.mh_prune_training_sample_memory()
        self.hypotheses = kept_hypotheses

        self.mh_training_sample_set.pop(pruned_frame_id, None)

    def mh_prune_random_frame_id(self):
        frame_id = random.sample(list(self.mh_training_sample_set),1)[0]
        self.mh_prune_frame_id(frame_id)

    def mh_print_summary(self):
        print("Current frame: ", self.frame_num)
        print([x.frame_ids for x in self.hypotheses])
        print([(x, self.mh_training_sample_set[x].num_supported_hypotheses) for x in self.mh_training_sample_set])

    def xs_update_memory(self, sample_x: TensorList, target_box, im_patch, learning_rate = None):
        """
        eXtremum Summary update memory, also handles random query case as well
        """
        # Update weights and get replace ind
        self.summary_updated = False
        self.query_requested = False

        sample = sample_x[0]
        summary_samples = self.training_samples[0][self.num_init_samples[0]:self.num_stored_samples[0],...]

        replace_ind = -1
        ignore_feedback = False

        # Randomly query for feedback with some set probability
        full_summary = self.num_stored_samples[0] - self.num_init_samples[0] >= self.summary_size[0]
        if self.params.get("fill_summary_first", False) and not full_summary:
            replace_ind = self.num_stored_samples[0] - self.num_init_samples[0]
            ignore_feedback = True

        elif self.params.get("use_rnd_query", False) and torch.rand(1) < self.params.get("random_query_probability", 0.05):
            # Keep if less than full, otherwise randomly replace sample
            if not full_summary:
                replace_ind = self.num_stored_samples[0] - self.num_init_samples[0]
            else:
                replace_ind = torch.randint(self.summary_size[0], (1,)).item()


        elif self.params.get("use_active_online_extremum", False):
            replace_ind, _, _, _ = kc.get_k_online_summary_update_index(summary_samples,
                                                                    sample, self.summary_size[0],
                                                                    threshold=self.extremum_summary_threshold,
                                                                    dist_func=self.params.get("dist_func","cosine_dist"),
                                                                    fill_first=self.params.get("fill_summary_first", False))

        if replace_ind > -1:
            # Use oracle for active learning, do it after extremum
            if self.params.get("use_oracle_feedback", False) and not ignore_feedback:
                self.query_requested = True

                # Throw out if the predicted bounding box is pretty far off
                pred_bbox = torch.cat((self.pos[[1, 0]] - (self.target_sz[[1, 0]] - 1) / 2, self.target_sz[[1, 0]]))
                oracle_iou_threshold = self.params.get('oracle_iou_threshold', 0.85)
                oracle_prec_threshold = self.params.get('oracle_prec_threshold', 5)

                gt = self.ground_truth_bbox
                iou = calc_iou_overlap(torch.tensor(gt).unsqueeze(0), pred_bbox.unsqueeze(0))

                if self.params.get('use_oracle_iou') and iou < oracle_iou_threshold:
                    return

            #print(f"Replacing ind: {replace_ind}")

            self.summary_updated = True

            self.training_samples[0][self.num_init_samples[0] + replace_ind, ...] = sample
            self.target_boxes[self.num_init_samples[0] + replace_ind, ...] = target_box
            img = im_patch.reshape(im_patch.size()[1:])
            img = torch.moveaxis((img - img.min()) / (img.max() - img.min()), 0, 2)
            self.summary_patches[replace_ind, ...] = img
            self.sample_weights[0][self.num_init_samples[0] + replace_ind] = self.sample_weights[0][0] # todo: make this a parameter

            if self.num_stored_samples[0] < self.num_init_samples[0] + self.summary_size[0]:
                self.num_stored_samples[0] += 1

            if self.params.get("use_active_online_extremum", False):
                if self.params.get("use_mean_score", True):
                    self.extremum_summary_threshold, _ = kc.get_mean_summary_score(self.training_samples[0][self.num_init_samples[0]:self.num_stored_samples[0], ...],
                        dist_func=self.params.get("dist_func","cosine_dist"))
                else:
                    self.extremum_summary_threshold *= self.params.get("summary_threshold_gamma", 1.005)

            num_summary_samples = self.num_stored_samples[0] - self.num_init_samples[0]

    def update_memory(self, sample_x: TensorList, target_box, im_patch, learning_rate = None):
        # Update weights and get replace ind
        replace_ind, summary_replace_ind = self.update_sample_weights(self.sample_weights, self.previous_replace_ind, self.num_stored_samples, self.num_init_samples, sample_x, learning_rate)
        self.previous_replace_ind = replace_ind
        # Update sample and label memory
        '''
        for train_samp, x, ind in zip(self.training_samples, sample_x, replace_ind):
            if (type(ind) == type(list())):
                train_samp[ind[0]:ind[0]+1,...] = x
            else:
                train_samp[ind:ind+1,...] = x
        '''

        # Update weights

        if summary_replace_ind != -1:
            # TODO: track mapping from frame_num to training_sample/summary_patches (note that training samples is all, summary_patches is only the new ones)
            self.training_samples[0][summary_replace_ind] = sample_x[0][0]

            self.target_boxes[summary_replace_ind,:] = target_box.to(self.params.device)
            img = im_patch.reshape(im_patch.size()[1:])
            img = torch.moveaxis((img-img.min())/(img.max()-img.min()), 0, 2)
            self.summary_patches[summary_replace_ind-self.num_init_samples[0]] = img
            self.summary_update = True
            self.summary_prev_ind = summary_replace_ind

            # Update and add hypotheses
            # Prune hypotheses of removed frames, should I do this?
            # TODO: do I actually ever use the frame_ids, can I get away with only mem_ids?
            # ANS: Yes, you will use it in the active setting, when you reach far back into history and prune things

            online_replace_ind = summary_replace_ind - self.num_init_samples[0]

            # Add new hypotheses and train them
            new_hypotheses = []
            for hypothesis in self.hypotheses:
                new_hypothesis = hypothesis.clone()
                new_hypothesis.frame_ids.add(self.frame_num)
                #new_hypothesis.mem_ids.add(summary_replace_ind)

                mem_ids = torch.tensor(list(new_hypothesis.mem_ids)).cuda()
                samples = torch.index_select(self.training_samples[0], 0, mem_ids)
                target_boxes = torch.index_select(self.target_boxes, 0, mem_ids) # self.target_boxes[:self.num_stored_samples[0], :].clone()
                sample_weights = torch.index_select(self.sample_weights[0], 0, mem_ids)  # self.sample_weights[0][:self.num_stored_samples[0]]

                num_iter = 2 # todo: fix this
                plot_loss = True

                # Run the filter optimizer module
                with torch.no_grad():
                    new_hypothesis.target_filter, _, losses = new_hypothesis.net.classifier.filter_optimizer(new_hypothesis.target_filter,
                                                                                                                num_iter=num_iter,
                                                                                                                feat=samples,
                                                                                                                bb=target_boxes,
                                                                                                                sample_weight=sample_weights,
                                                                                                                compute_losses=plot_loss)
                # todo: this seems dumb
                new_hypotheses.append(hypothesis)
                new_hypotheses.append(new_hypothesis)
            self.hypotheses = new_hypotheses

        # Update bb memory
        #self.target_boxes[replace_ind[0],:] = target_box.to(self.params.device)

        print("Number of tracks", len(self.hypotheses))
        # TODO: why does this not stop going up?
        self.num_stored_samples[0] += 1

    def update_sample_weights(self, sample_weights, previous_replace_ind, num_stored_samples, num_init_samples, sample_x = None, learning_rate = None):
        # Update weights and get index to replace
        #import pdb; pdb.set_trace()
        replace_ind = -1

        s_ind = self.sample_memory_size
        summary_replace_ind = -1
        observation = sample_x[0]
        # UPDATE_HERE
        new_threshold = 0

        self.in_default = False

        # If summary buffer is not full yet, simply add
        if self.summary_stored_samples < self.summary_size[0]:
            summary_replace_ind = self.summary_stored_samples
            self.summary_stored_samples = self.summary_stored_samples + 1
            self.in_default = True
        else:
            extremum_summary_set = self.training_samples[0][num_init_samples[0]:s_ind]
            if self.del_summary_num != -1:
                summary_replace_ind = self.del_summary_num
                self.del_summary_num = -1
            else:
                if self.use_saved_threshold:
                    summary_replace_ind, new_threshold = kc.online_summary_update_index(extremum_summary_set, observation, self.params.summary_size, threshold = self.saved_threshold)
                else:
                    summary_replace_ind, new_threshold = kc.online_summary_update_index(extremum_summary_set, observation, self.params.summary_size)
            #print(f"Threshold is {threshold}")

        #print(f"The summary replace index is {summary_replace_ind}")
        #issue is self.saved_threshold begins at 0

        if self.lock_mode:
            summary_replace_ind = -1

        if not self.in_default and summary_replace_ind != -1:
            #summary_replace_ind = -1
            '''
            new_summary_set = self.training_samples[0][num_init_samples[0]:s_ind]
            new_summary_set[summary_replace_ind] = observation
            new_threshold = kc.threshold_cost(new_summary_set)
            if new_threshold < self.use_saved_threshold:
                summary_replace_ind = -1 '''
            pass

        if summary_replace_ind != -1:
            if not self.in_default:
                self.use_saved_threshold = False
                self.saved_threshold = new_threshold
            summary_replace_ind = summary_replace_ind + num_init_samples[0]
        #print(f"saved threshold is {self.saved_threshold}")
        return replace_ind, summary_replace_ind

    def update_state(self, new_pos, new_scale = None):
        # Update scale
        if new_scale is not None:
            self.target_scale = new_scale.clamp(self.min_scale_factor, self.max_scale_factor)
            self.target_sz = self.base_target_sz * self.target_scale

        # Update pos
        inside_ratio = self.params.get('target_inside_ratio', 0.2)
        inside_offset = (inside_ratio - 0.5) * self.target_sz
        self.pos = torch.max(torch.min(new_pos, self.image_sz - inside_offset), inside_offset)

    def get_iounet_box(self, pos, sz, sample_pos, sample_scale):
        """All inputs in original image coordinates.
        Generates a box in the cropped image sample reference frame, in the format used by the IoUNet."""
        box_center = (pos - sample_pos) / sample_scale + (self.img_sample_sz - 1) / 2
        box_sz = sz / sample_scale
        target_ul = box_center - (box_sz - 1) / 2
        return torch.cat([target_ul.flip((0,)), box_sz.flip((0,))])

    def init_iou_net(self, backbone_feat):
        # Setup IoU net and objective
        for p in self.net.bb_regressor.parameters():
            p.requires_grad = False

        # Get target boxes for the different augmentations
        self.classifier_target_box = self.get_iounet_box(self.pos, self.target_sz, self.init_sample_pos, self.init_sample_scale)
        target_boxes = TensorList()
        if self.params.iounet_augmentation:
            for T in self.transforms:
                if not isinstance(T, (augmentation.Identity, augmentation.Translation, augmentation.FlipHorizontal, augmentation.FlipVertical, augmentation.Blur)):
                    break
                target_boxes.append(self.classifier_target_box + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
        else:
            target_boxes.append(self.classifier_target_box + torch.Tensor([self.transforms[0].shift[1], self.transforms[0].shift[0], 0, 0]))
        target_boxes = torch.cat(target_boxes.view(1,4), 0).to(self.params.device)

        # Get iou features
        iou_backbone_feat = self.get_iou_backbone_features(backbone_feat)

        # Remove other augmentations such as rotation
        iou_backbone_feat = TensorList([x[:target_boxes.shape[0],...] for x in iou_backbone_feat])

        # Get modulation vector
        self.iou_modulation = self.get_iou_modulation(iou_backbone_feat, target_boxes)
        if torch.is_tensor(self.iou_modulation[0]):
            self.iou_modulation = TensorList([x.detach().mean(0) for x in self.iou_modulation])


    def init_classifier(self, init_backbone_feat):
        # Get classification features
        x = self.get_classification_features(init_backbone_feat)

        # Overwrite some parameters in the classifier. (These are not generally changed)
        self._overwrite_classifier_params(feature_dim=x.shape[-3])

        # Add the dropout augmentation here, since it requires extraction of the classification features
        if 'dropout' in self.params.augmentation and self.params.get('use_augmentation', True):
            num, prob = self.params.augmentation['dropout']
            self.transforms.extend(self.transforms[:1]*num)
            x = torch.cat([x, F.dropout2d(x[0:1,...].expand(num,-1,-1,-1), p=prob, training=True)])

        # Set feature size and other related sizes
        self.feature_sz = torch.Tensor(list(x.shape[-2:]))
        ksz = self.net.classifier.filter_size
        self.kernel_size = torch.Tensor([ksz, ksz] if isinstance(ksz, (int, float)) else ksz)
        self.output_sz = self.feature_sz + (self.kernel_size + 1)%2

        # Construct output window
        self.output_window = None
        if self.params.get('window_output', False):
            if self.params.get('use_clipped_window', False):
                self.output_window = dcf.hann2d_clipped(self.output_sz.long(), (self.output_sz*self.params.effective_search_area / self.params.search_area_scale).long(), centered=True).to(self.params.device)
            else:
                self.output_window = dcf.hann2d(self.output_sz.long(), centered=True).to(self.params.device)
            self.output_window = self.output_window.squeeze(0)

        # Get target boxes for the different augmentations
        #import pdb; pdb.set_trace()
        target_boxes = self.init_target_boxes()
        #note that after the 13 augmentations, the transformation performed is the identity

        # Set number of iterations
        plot_loss = self.params.debug > 0
        num_iter = self.params.get('net_opt_iter', None)

        # Get target filter by running the discriminative model prediction module
        with torch.no_grad():
            self.target_filter, _, losses = self.net.classifier.get_filter(x, target_boxes, num_iter=num_iter,
                                                                           compute_losses=plot_loss)

        # Init memory
        if self.params.get('update_classifier', True):
            self.init_memory(TensorList([x]))
        # at this point, the k_center_summary is initialized along with the initial augmentations
        if plot_loss:
            if isinstance(losses, dict):
                losses = losses['train']
            self.losses = torch.cat(losses)
            if self.visdom is not None:
                self.visdom.register((self.losses, torch.arange(self.losses.numel())), 'lineplot', 3, 'Training Loss' + self.id_str)
            elif self.params.debug >= 3:
                plot_graph(self.losses, 10, title='Training Loss' + self.id_str)

    def _overwrite_classifier_params(self, feature_dim):
        # Overwrite some parameters in the classifier. (These are not generally changed)
        pred_module = getattr(self.net.classifier.filter_optimizer, 'score_predictor', self.net.classifier.filter_optimizer)
        if self.params.get('label_threshold', None) is not None:
            self.net.classifier.filter_optimizer.label_threshold = self.params.label_threshold
        if self.params.get('label_shrink', None) is not None:
            self.net.classifier.filter_optimizer.label_shrink = self.params.label_shrink
        if self.params.get('softmax_reg', None) is not None:
            self.net.classifier.filter_optimizer.softmax_reg = self.params.softmax_reg
        if self.params.get('filter_reg', None) is not None:
            pred_module.filter_reg[0] = self.params.filter_reg
            pred_module.min_filter_reg = self.params.filter_reg
        if self.params.get('filter_init_zero', False):
            self.net.classifier.filter_initializer = FilterInitializerZero(self.net.classifier.filter_size, feature_dim)

    def update_classifier(self, train_x, target_box, im_patch, learning_rate=None, scores=None):
        # Set flags and learning rate
        hard_negative_flag = learning_rate is not None
        if learning_rate is None:
            learning_rate = self.params.learning_rate

        self.summary_updated = False
        self.query_requested = False

        # Update the tracker memory
        if hard_negative_flag or self.frame_num % self.params.get('train_sample_interval', 1) == 0:
            if self.params.get("use_active_online_summary", False):
                self.xs_update_memory(TensorList([train_x]), target_box, im_patch, learning_rate)
            elif self.params.get("use_mh", False) and self.params.get('use_extremum_pruning', False):
                self.mh_update_memory_extremum(TensorList([train_x]), target_box, im_patch, learning_rate)
            elif self.params.get("use_mh", False):
                self.mh_update_memory(TensorList([train_x]), target_box, im_patch, learning_rate)
            else:
                self.update_memory(TensorList([train_x]), target_box, im_patch, learning_rate)

        # Decide the number of iterations to run
        num_iter = 0
        low_score_th = self.params.get('low_score_opt_threshold', None)
        if hard_negative_flag:
            num_iter = self.params.get('net_opt_hn_iter', None)
        elif low_score_th is not None and low_score_th > scores.max().item():
            num_iter = self.params.get('net_opt_low_iter', None)
        elif self.summary_update:
            num_iter = self.params.get('summary_update_num_iter', 2)
            #print(f"Num iter is {num_iter} due to summary update")
        elif (self.frame_num - 1) % self.params.train_skipping == 0:
            num_iter = self.params.get('net_opt_update_iter', None)
            #print(f"Num iter {num_iter} due to every 20 updates")

        plot_loss = self.params.debug > 0

        # Multi-hypothesis has its own way of updating all the filters
        if self.params.get("use_mh", False):
            num_iter = 0

        if num_iter > 0:
            # Get inputs for the DiMP filter optimizer module
            samples = self.training_samples[0][:self.num_stored_samples[0],...]
            target_boxes = self.target_boxes[:self.num_stored_samples[0],:].clone()
            sample_weights = self.sample_weights[0][:self.num_stored_samples[0]]

            # Run the filter optimizer module
            with torch.no_grad():
                self.target_filter, _, losses = self.net.classifier.filter_optimizer(self.target_filter,
                                                                                     num_iter=num_iter, feat=samples,
                                                                                     bb=target_boxes,
                                                                                     sample_weight=sample_weights,
                                                                                     compute_losses=plot_loss)
            if plot_loss:
                if isinstance(losses, dict):
                    losses = losses['train']
                self.losses = torch.cat((self.losses, torch.cat(losses)))
                if self.visdom is not None:
                    self.visdom.register((self.losses, torch.arange(self.losses.numel())), 'lineplot', 3, 'Training Loss' + self.id_str)
                elif self.params.debug >= 3:
                    plot_graph(self.losses, 10, title='Training Loss' + self.id_str)

    def refine_target_box(self, backbone_feat, sample_pos, sample_scale, scale_ind, update_scale = True):
        """Run the ATOM IoUNet to refine the target bounding box."""

        if hasattr(self.net.bb_regressor, 'predict_bb'):
            return self.direct_box_regression(backbone_feat, sample_pos, sample_scale, scale_ind, update_scale)

        # Initial box for refinement
        init_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos, sample_scale)

        # Extract features from the relevant scale
        iou_features = self.get_iou_features(backbone_feat)
        iou_features = TensorList([x[scale_ind:scale_ind+1,...] for x in iou_features])

        # Generate random initial boxes
        init_boxes = init_box.view(1,4).clone()
        if self.params.num_init_random_boxes > 0:
            square_box_sz = init_box[2:].prod().sqrt()
            rand_factor = square_box_sz * torch.cat([self.params.box_jitter_pos * torch.ones(2), self.params.box_jitter_sz * torch.ones(2)])

            minimal_edge_size = init_box[2:].min()/3
            rand_bb = (torch.rand(self.params.num_init_random_boxes, 4) - 0.5) * rand_factor
            new_sz = (init_box[2:] + rand_bb[:,2:]).clamp(minimal_edge_size)
            new_center = (init_box[:2] + init_box[2:]/2) + rand_bb[:,:2]
            init_boxes = torch.cat([new_center - new_sz/2, new_sz], 1)
            init_boxes = torch.cat([init_box.view(1,4), init_boxes])

        # Optimize the boxes
        output_boxes, output_iou = self.optimize_boxes(iou_features, init_boxes)

        # Remove weird boxes
        output_boxes[:, 2:].clamp_(1)
        aspect_ratio = output_boxes[:,2] / output_boxes[:,3]
        keep_ind = (aspect_ratio < self.params.maximal_aspect_ratio) * (aspect_ratio > 1/self.params.maximal_aspect_ratio)
        output_boxes = output_boxes[keep_ind,:]
        output_iou = output_iou[keep_ind]

        # If no box found
        if output_boxes.shape[0] == 0:
            return

        # Predict box
        k = self.params.get('iounet_k', 5)
        topk = min(k, output_boxes.shape[0])
        _, inds = torch.topk(output_iou, topk)
        predicted_box = output_boxes[inds, :].mean(0)
        predicted_iou = output_iou.view(-1, 1)[inds, :].mean(0)

        # Get new position and size
        new_pos = predicted_box[:2] + predicted_box[2:] / 2
        new_pos = (new_pos.flip((0,)) - (self.img_sample_sz - 1) / 2) * sample_scale + sample_pos
        new_target_sz = predicted_box[2:].flip((0,)) * sample_scale
        new_scale = torch.sqrt(new_target_sz.prod() / self.base_target_sz.prod())

        self.pos_iounet = new_pos.clone()

        if self.params.get('use_iounet_pos_for_learning', True):
            self.pos = new_pos.clone()

        self.target_sz = new_target_sz

        if update_scale:
            self.target_scale = new_scale

        # self.visualize_iou_pred(iou_features, predicted_box)


    def optimize_boxes(self, iou_features, init_boxes):
        box_refinement_space = self.params.get('box_refinement_space', 'default')
        if box_refinement_space == 'default':
            return self.optimize_boxes_default(iou_features, init_boxes)
        if box_refinement_space == 'relative':
            return self.optimize_boxes_relative(iou_features, init_boxes)
        raise ValueError('Unknown box_refinement_space {}'.format(box_refinement_space))


    def optimize_boxes_default(self, iou_features, init_boxes):
        """Optimize iounet boxes with the default parametrization"""
        output_boxes = init_boxes.view(1, -1, 4).to(self.params.device)
        step_length = self.params.box_refinement_step_length
        if isinstance(step_length, (tuple, list)):
            step_length = torch.Tensor([step_length[0], step_length[0], step_length[1], step_length[1]], device=self.params.device).view(1,1,4)

        for i_ in range(self.params.box_refinement_iter):
            # forward pass
            bb_init = output_boxes.clone().detach()
            bb_init.requires_grad = True

            outputs = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, bb_init)

            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            outputs.backward(gradient = torch.ones_like(outputs))

            # Update proposal
            output_boxes = bb_init + step_length * bb_init.grad * bb_init[:, :, 2:].repeat(1, 1, 2)
            output_boxes.detach_()

            step_length *= self.params.box_refinement_step_decay

        return output_boxes.view(-1,4).cpu(), outputs.detach().view(-1).cpu()


    def optimize_boxes_relative(self, iou_features, init_boxes):
        """Optimize iounet boxes with the relative parametrization ised in PrDiMP"""
        output_boxes = init_boxes.view(1, -1, 4).to(self.params.device)
        step_length = self.params.box_refinement_step_length
        if isinstance(step_length, (tuple, list)):
            step_length = torch.Tensor([step_length[0], step_length[0], step_length[1], step_length[1]]).to(self.params.device).view(1,1,4)

        sz_norm = output_boxes[:,:1,2:].clone()
        output_boxes_rel = bbutils.rect_to_rel(output_boxes, sz_norm)
        for i_ in range(self.params.box_refinement_iter):
            # forward pass
            bb_init_rel = output_boxes_rel.clone().detach()
            bb_init_rel.requires_grad = True

            bb_init = bbutils.rel_to_rect(bb_init_rel, sz_norm)
            outputs = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, bb_init)

            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            outputs.backward(gradient = torch.ones_like(outputs))

            # Update proposal
            output_boxes_rel = bb_init_rel + step_length * bb_init_rel.grad
            output_boxes_rel.detach_()

            step_length *= self.params.box_refinement_step_decay

        #     for s in outputs.view(-1):
        #         print('{:.2f}  '.format(s.item()), end='')
        #     print('')
        # print('')

        output_boxes = bbutils.rel_to_rect(output_boxes_rel, sz_norm)

        return output_boxes.view(-1,4).cpu(), outputs.detach().view(-1).cpu()

    def direct_box_regression(self, backbone_feat, sample_pos, sample_scale, scale_ind, update_scale = True):
        """Implementation of direct bounding box regression."""

        # Initial box for refinement
        init_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos, sample_scale)

        # Extract features from the relevant scale
        iou_features = self.get_iou_features(backbone_feat)
        iou_features = TensorList([x[scale_ind:scale_ind+1,...] for x in iou_features])

        # Generate random initial boxes
        init_boxes = init_box.view(1, 1, 4).clone().to(self.params.device)

        # Optimize the boxes
        output_boxes = self.net.bb_regressor.predict_bb(self.iou_modulation, iou_features, init_boxes).view(-1,4).cpu()

        # Remove weird boxes
        output_boxes[:, 2:].clamp_(1)

        predicted_box = output_boxes[0, :]

        # Get new position and size
        new_pos = predicted_box[:2] + predicted_box[2:] / 2
        new_pos = (new_pos.flip((0,)) - (self.img_sample_sz - 1) / 2) * sample_scale + sample_pos
        new_target_sz = predicted_box[2:].flip((0,)) * sample_scale
        new_scale_bbr = torch.sqrt(new_target_sz.prod() / self.base_target_sz.prod())
        new_scale = new_scale_bbr

        self.pos_iounet = new_pos.clone()

        if self.params.get('use_iounet_pos_for_learning', True):
            self.pos = new_pos.clone()

        self.target_sz = new_target_sz

        if update_scale:
            self.target_scale = new_scale


    def visualize_iou_pred(self, iou_features, center_box):
        center_box = center_box.view(1,1,4)
        sz_norm = center_box[...,2:].clone()
        center_box_rel = bbutils.rect_to_rel(center_box, sz_norm)

        pos_dist = 1.0
        sz_dist = math.log(3.0)
        pos_step = 0.01
        sz_step = 0.01

        pos_scale = torch.arange(-pos_dist, pos_dist+pos_step, step=pos_step)
        sz_scale = torch.arange(-sz_dist, sz_dist+sz_step, step=sz_step)

        bbx = torch.zeros(1, pos_scale.numel(), 4)
        bbx[0,:,0] = pos_scale.clone()
        bby = torch.zeros(pos_scale.numel(), 1, 4)
        bby[:,0,1] = pos_scale.clone()
        bbw = torch.zeros(1, sz_scale.numel(), 4)
        bbw[0,:,2] = sz_scale.clone()
        bbh = torch.zeros(sz_scale.numel(), 1, 4)
        bbh[:,0,3] = sz_scale.clone()

        pos_boxes = bbutils.rel_to_rect((center_box_rel + bbx) + bby, sz_norm).view(1,-1,4).to(self.params.device)
        sz_boxes = bbutils.rel_to_rect((center_box_rel + bbw) + bbh, sz_norm).view(1,-1,4).to(self.params.device)

        pos_scores = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, pos_boxes).exp()
        sz_scores = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, sz_boxes).exp()

        show_tensor(pos_scores.view(pos_scale.numel(),-1), title='Position scores', fig_num=21)
        show_tensor(sz_scores.view(sz_scale.numel(),-1), title='Size scores', fig_num=22)


    def visdom_draw_tracking(self, image, box, segmentation=None):
        if hasattr(self, 'search_area_box'):
            self.visdom.register((image, box, self.search_area_box), 'Tracking', 1, 'Tracking')
        else:
            self.visdom.register((image, box), 'Tracking', 1, 'Tracking')

    def return_summary_patches(self):
        if self.params.get("use_mh", True):
            return [self.mh_training_sample_set[frame_id].im_patch for frame_id in self.mh_training_sample_set]
        else:
            return self.summary_patches
            #return self.ref_patches[:self.summary_size[0]]

    def return_summary_bbox(self):
        if self.params.get("use_mh", True):
            return [self.mh_training_sample_set[frame_id].target_box for frame_id in self.mh_training_sample_set]
        else:
            start_ind = self.num_init_samples[0]
            end_ind = self.online_start_ind[0]
            return self.target_boxes[start_ind:end_ind]
            #return self.target_boxes[:self.summary_size[0]]

    def return_bbox(self):
        return self.target_boxes

    def delete_summary_num(self, index):
        #convert the index to the actual index in memory

        # above step not needed
        self.del_summary_num = index
    def update_bbox(self, index, bbox):
        adj_index = index + self.num_init_samples[0]
        self.target_boxes[adj_index] = bbox

    def set_lock_true(self):
        self.lock_mode = True

    def set_lock_false(self):
        self.lock_mode = False
