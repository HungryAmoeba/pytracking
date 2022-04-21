from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone

def parameters():
    params = TrackerParams()

    # Visualization/debug params
    params.debug = 1
    params.visualization = True
    params.interactive = True

    params.use_gpu = True

    # Bandwidth params
    params.use_limited_bandwidth = True
    params.num_frames_between_queries = 45

    # Interactivity params
    params.use_interaction = True
    params.query_policy = "summary_threshold" # "summary_threshold", "fixed_rate"
    params.train_from_previous = True

    # Oracle parameters
    params.use_oracle_feedback = True
    params.use_oracle_iou = True
    params.use_oracle_prec = False
    params.oracle_iou_threshold = 0.7
    params.oracle_prec_threshold = 5

    # Global trainer params
    params.skip_update_classifier = False
    params.use_global_trainer = False
    params.use_global_extremum = False

    # Extremum summary params
    params.use_summary = True
    params.use_summary_update = False
    params.summary_size = 15 # Currently useless, set by number of initial samples
    params.log_summary = True # this is only for data saving at the moment
    params.summary_replacement_policy = "extremum" # "extremum", "random"
    params.threshold_update_policy = "constant" # "mean_score", "constant"
    params.threshold_initialization_policy = "mean_score" # "mean_score", "constant"
    params.fill_summary_first = True
    params.summary_threshold_update_constant = 1.01
    params.default_summary_threshold = 0.0
    params.dist_func = "cosine_dist" # "cosine_dist", "l2_normalised_dist", "l2_dist"
    params.summary_rel_weight = 1

    # DiMP parameters
    params.image_sample_size = 22*16
    params.search_area_scale = 6
    params.border_mode = 'inside_major'
    params.patch_max_scale_change = 1.5

    # Learning parameters
    params.sample_memory_size = 50 #this gets overwritten in init_target_boxes and is obsolete
    params.learning_rate = 0.01
    params.init_samples_minimum_weight = 0.25
    params.use_continuous_training = False
    params.train_skipping = 20

    # Net optimization params
    params.update_classifier = True
    params.net_opt_iter = 10
    params.net_opt_update_iter = 2
    params.net_opt_hn_iter = 1

    # Detection parameters
    params.window_output = False

    # Init augmentation parameters
    params.use_augmentation = True
    params.augmentation = {'fliplr': True,
                           'rotate': [10, -10, 45, -45],
                           'blur': [(3,1), (1, 3), (2, 2)],
                           'relativeshift': [(0.6, 0.6), (-0.6, 0.6), (0.6, -0.6), (-0.6,-0.6)],
                           'dropout': (2, 0.2)}

    params.augmentation_expansion_factor = 2
    params.random_shift_factor = 1/3

    # Advanced localization parameters
    params.advanced_localization = True
    params.target_not_found_threshold = 0.25
    params.distractor_threshold = 0.8
    params.hard_negative_threshold = 0.5
    params.target_neighborhood_scale = 2.2
    params.dispalcement_scale = 0.8
    params.hard_negative_learning_rate = 0.02
    params.update_scale_when_uncertain = True

    # IoUnet parameters
    params.box_refinement_space = 'relative'
    params.iounet_augmentation = False      # Use the augmented samples to compute the modulation vector
    params.iounet_k = 3                     # Top-k average to estimate final box
    params.num_init_random_boxes = 9        # Num extra random boxes in addition to the classifier prediction
    params.box_jitter_pos = 0.1             # How much to jitter the translation for random boxes
    params.box_jitter_sz = 0.5              # How much to jitter the scale for random boxes
    params.maximal_aspect_ratio = 6         # Limit on the aspect ratio
    params.box_refinement_iter = 10          # Number of iterations for refining the boxes
    params.box_refinement_step_length = 2.5e-3 # 1   # Gradient step length in the bounding box refinement
    params.box_refinement_step_decay = 1    # Multiplicative step length decay (1 means no decay)

    params.net = NetWithBackbone(net_path='super_dimp.pth.tar',
                                 use_gpu=params.use_gpu)

    params.vot_anno_conversion_type = 'preserve_area'

    return params
