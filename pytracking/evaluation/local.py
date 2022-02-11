from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_path = '/home/cail/data/LaSOTBenchmark'
    settings.network_path = '/home/cail/Documents/pytracking_summary/pytracking/networks/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = '/home/cail/data/OTB100'
    settings.result_plot_path = '/home/cail/Documents/pytracking_summary/pytracking/result_plots/'
    settings.results_path = '/home/cail/Documents/pytracking_summary/pytracking/tracking_results/'    # Where to store tracking results
    settings.segmentation_path = '/home/cail/Documents/pytracking_summary/pytracking/segmentation_results/'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = '/home/cail/data/Dataset_UAV123/UAV123'
    settings.vot_path = '/home/cail/data/VOT/VOT_2018/sequences'
    settings.vot_2020_path = '/home/cail/data/VOT/VOT_2020/sequences'
    settings.youtubevos_dir = ''
    settings.fish_path = '/home/cail/Documents/mad-toolkit/data/MADv0_ST'

    return settings

