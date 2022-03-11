from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_path = '/home/warp/Documents/LaSOTTesting'
    settings.network_path = '/home/cail/Documents/pytracking_summary/pytracking/networks/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = '/home/warp/Documents/otb100'
    settings.result_plot_path = '/home/cail/Documents/pytracking_summary/pytracking/result_plots/'
    settings.results_path = '/home/cail/Documents/pytracking_summary/pytracking/tracking_results/'    # Where to store tracking results
    settings.segmentation_path = '/home/warp/Documents/Charles/pytracking/pytracking/segmentation_results/'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = '/home/warp/Documents/UAV123/'
    settings.vot_path = '/home/warp/Documents/Charles/vot-workspace/sequences'
    settings.youtubevos_dir = ''
    settings.fish_path = '/home/cail/Documents/mad-toolkit/data/MADv0_ST'
    settings.bagging_dir = '/home/warp/Documents/Charles/pytracking/pytracking/bagged_summary_images'
    settings.vot_2020_path = '/home/warp/Documents/Charles/vot-2020ST/sequences'

    return settings
