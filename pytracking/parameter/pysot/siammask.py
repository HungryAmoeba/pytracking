from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone

def parameters():
    params = TrackerParams()

    params.path_to_pysot_install = "/home/cail/Documents/pysot"
    params.config_file = "/home/cail/Documents/pysot/experiments/siammask_r50_l3/config.yaml"
    params.model_file = "/home/cail/Documents/pysot/experiments/siammask_r50_l3/model.pth"

    return params