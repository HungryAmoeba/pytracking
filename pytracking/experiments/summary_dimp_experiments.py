from pytracking.evaluation import Tracker, get_dataset, trackerlist

def run_all_datasets_5_replicates():
    trackers = trackerlist('dimp_original', 'super_dimp', range(5)) + \
               trackerlist('dimp_summary', 'super_dimp_15', range(5))

    dataset = get_dataset('uav', 'lasot', 'otb', 'nfs', 'vot', 'vot_2020', 'fish')

    return trackers, dataset

def run_all_datasets_3_replicates():
    trackers = trackerlist('dimp_original', 'super_dimp', range(3)) + \
               trackerlist('dimp_summary', 'super_dimp_15', range(3))

    dataset = get_dataset('uav', 'lasot', 'otb', 'nfs', 'vot', 'vot_2020', 'fish')

    return trackers, dataset

def UAV123_test():
    trackers = trackerlist('dimp_original', 'super_dimp', range(5)) + \
               trackerlist('dimp_summary', 'super_dimp_15', range(5))

    dataset = get_dataset('uav')

    return trackers, dataset

def OTB_test():
    trackers = trackerlist('dimp_original', 'super_dimp', range(5))

    dataset = get_dataset('otb')

    return trackers, dataset

def LaSOT_test():
    trackers = trackerlist('dimp_original', 'super_dimp', range(5)) + \
               trackerlist('dimp_summary', 'super_dimp_15', range(5))

    dataset = get_dataset('lasot')

    return trackers, dataset

def NFS_test():
    trackers = trackerlist('dimp_original', 'super_dimp', range(5)) + \
               trackerlist('dimp_summary', 'super_dimp_15', range(5))

    dataset = get_dataset('nfs')

    return trackers, dataset

def VOT2018_test():
    trackers = trackerlist('dimp_original', 'super_dimp', range(5)) + \
               trackerlist('dimp_summary', 'super_dimp_15', range(5))

    dataset = get_dataset('vot')

    return trackers, dataset

def VOT2020_test():
    trackers = trackerlist('dimp_original', 'super_dimp', range(5)) + \
               trackerlist('dimp_summary', 'super_dimp_15', range(5))

    dataset = get_dataset('vot_2020')

    return trackers, dataset

def concept_drift_test():
    trackers = trackerlist('dimp_original', 'super_dimp_baseline_only', range(5)) + \
               trackerlist('dimp_original', 'super_dimp', range(5)) + \
               trackerlist('dimp_original', 'super_dimp_online_only', range(5)) + \
               trackerlist('dimp_original', 'super_dimp_online_only_10', range(5)) + \
               trackerlist('dimp_original', 'super_dimp_online_only_1', range(5))
    dataset = get_dataset('fish')

    return trackers, dataset

def global_summary_fish_test():

    trackers = trackerlist('mh_dimp_summary', 'super_dimp_global_x_mean_l2', range(5)) + \
               trackerlist('mh_dimp_summary', 'super_dimp_global_x_mean_cd', range(5)) + \
               trackerlist('mh_dimp_summary', 'super_dimp_global_x_gamma1_01_l2', range(5)) + \
               trackerlist('mh_dimp_summary', 'super_dimp_global_x_gamma1_01_cd', range(5)) + \
               trackerlist('mh_dimp_summary', 'super_dimp_global_x_gamma1_005_l2', range(5)) + \
               trackerlist('mh_dimp_summary', 'super_dimp_global_x_gamma1_005_cd', range(5))
    dataset = get_dataset('fish')
    return trackers, dataset

def online_summary_fish_test():

    trackers = trackerlist('mh_dimp_summary', 'super_dimp_online_x_mean_l2', range(5)) + \
               trackerlist('mh_dimp_summary', 'super_dimp_online_x_mean_cd', range(5)) + \
               trackerlist('mh_dimp_summary', 'super_dimp_online_x_gamma_005_l2', range(5)) + \
               trackerlist('mh_dimp_summary', 'super_dimp_online_x_gamma_005_cd', range(5))
    dataset = get_dataset('fish')
    return trackers, dataset

def online_fill_first_fish_test():

    trackers = trackerlist('mh_dimp_summary', 'super_dimp_online_random_3_fill_first', range(5)) + \
               trackerlist('mh_dimp_summary', 'super_dimp_online_x_mean_cd_fill_first', range(5)) + \
               trackerlist('mh_dimp_summary', 'super_dimp_online_random_03_fill_first', range(5)) + \
               trackerlist('mh_dimp_summary', 'super_dimp_online_x_gamma_005_cd_fill_first', range(5))
    dataset = get_dataset('fish', 'vot_2020')
    return trackers, dataset

def original_dimp_all_minus_lasot_test():

    trackers = trackerlist('dimp_original', 'super_dimp', range(5))
    dataset = get_dataset('fish', 'vot_2020', 'uav', 'otb', 'nfs')
    return trackers, dataset

def original_dimp_lasot_test():

    trackers = trackerlist('dimp_original', 'super_dimp', range(5))
    dataset = get_dataset('lasot')
    return trackers, dataset

def xs_random_fish_test():

    trackers = trackerlist('mh_dimp_summary', 'online_rand_003_ff', range(5)) + \
               trackerlist('mh_dimp_summary', 'online_rand_03_ff', range(5)) + \
               trackerlist('mh_dimp_summary', 'online_rand_3_ff', range(5))
    dataset = get_dataset('fish')
    return trackers, dataset

def xs_gamma_fish_test():

    trackers = trackerlist('mh_dimp_summary', 'online_gamma_005_cd_ff', range(5)) + \
               trackerlist('mh_dimp_summary', 'online_gamma_05_cd_ff', range(5)) + \
               trackerlist('mh_dimp_summary', 'online_gamma_5_cd_ff', range(5))
    dataset = get_dataset('fish')
    return trackers, dataset

def xs_mean_dist_fish_test():

    trackers = trackerlist('mh_dimp_summary', 'online_mean_l2_ff', range(5)) + \
               trackerlist('mh_dimp_summary', 'online_mean_cd_ff', range(5))
    dataset = get_dataset('fish')
    return trackers, dataset

def xs_mean_lasot_test():

    trackers = trackerlist('mh_dimp_summary', 'online_mean_cd_ff', range(5))
    dataset = get_dataset('lasot')
    return trackers, dataset

def xs_mean_all_minus_lasot_test():

    trackers = trackerlist('mh_dimp_summary', 'online_mean_cd_ff', range(5))
    dataset = get_dataset('vot_2020', 'uav', 'otb')
    return trackers, dataset

def online_summary_test():

    trackers = trackerlist('mh_dimp_summary', 'super_dimp_online_x_gamma_005_l2', range(5)) + \
               trackerlist('mh_dimp_summary', 'super_dimp_online_x_gamma_005_cd', range(5)) + \
               trackerlist('dimp_original', 'super_dimp', range(5))
    dataset = get_dataset('vot_2020', 'otb', 'uav')
    return trackers, dataset

def online_summary_lasot_test():

    trackers = trackerlist('mh_dimp_summary', 'super_dimp_online_x_gamma_005_l2', range(5)) + \
               trackerlist('mh_dimp_summary', 'super_dimp_online_x_gamma_005_cd', range(5)) + \
               trackerlist('dimp_original', 'super_dimp', range(5))
    dataset = get_dataset('lasot')
    return trackers, dataset

def xs_v_rand_always_query_fish_test():
    trackers = trackerlist('mh_dimp_summary', 'online_gamma_0_default_0_0_cd_ff', range(5)) + \
               trackerlist('mh_dimp_summary', 'online_rand_always_ff', range(5))

    dataset = get_dataset('fish')
    return trackers, dataset

def mh_fish_test():
    #trackers = trackerlist('dimp_original', 'super_dimp_online_only_10', range(1))
    trackers = trackerlist('mh_dimp_summary', 'online_gamma_0_default_0_0_cd_ff', range(1))
    #trackers = trackerlist('mh_dimp_summary', 'online_gamma_0_default_0_0_cd_ff', range(1)) + \
    #           trackerlist('mh_dimp_summary', 'online_gamma_0_default_0_8_cd_ff', range(1))
    # trackers = trackerlist('mh_dimp_summary', 'super_dimp_15', range(1)) + \
    #            trackerlist('dimp_original', 'super_dimp', range(1)) + \
    #            trackerlist('mh_dimp_summary', 'super_dimp_baseline_set', range(1))

    dataset = get_dataset('fish')

    return trackers, dataset

def bl_fish_test():
    #trackers = trackerlist('bl_dimp_summary', 'online_mean_cd_thresholded_bl_5', range(1))
    trackers = trackerlist('bl_dimp_summary', 'online_mean_cd_thresholded_bl_60', range(5)) + \
               trackerlist('bl_dimp_summary', 'online_mean_cd_thresholded_bl_30', range(5)) + \
               trackerlist('bl_dimp_summary', 'online_mean_cd_thresholded_bl_10', range(5)) + \
               trackerlist('bl_dimp_summary', 'online_mean_cd_thresholded_bl_1', range(5)) + \
               trackerlist('bl_dimp_summary', 'online_mean_cd_thresholded_bl_100', range(5)) + \
               trackerlist('bl_dimp_summary', 'online_rand_bl_60', range(5)) + \
               trackerlist('bl_dimp_summary', 'online_rand_bl_30', range(5)) + \
               trackerlist('bl_dimp_summary', 'online_rand_bl_10', range(5)) + \
               trackerlist('bl_dimp_summary', 'online_rand_bl_100', range(5)) + \
               trackerlist('bl_dimp_summary', 'online_rand_bl_1', range(5)) + \
               trackerlist('bl_dimp_summary', 'online_xs_cd_bl_1', range(5)) + \
               trackerlist('bl_dimp_summary', 'online_xs_cd_bl_10', range(5)) + \
               trackerlist('bl_dimp_summary', 'online_xs_cd_bl_30', range(5)) + \
               trackerlist('bl_dimp_summary', 'online_xs_cd_bl_60', range(5)) + \
               trackerlist('bl_dimp_summary', 'online_xs_cd_bl_100', range(5))
    dataset = get_dataset('fish')

    return trackers, dataset

def bl_thresholded_query_fish_test():
    trackers = trackerlist('bl_dimp_summary', 'online_mean_cd_thresholded_bl_60', range(5)) + \
               trackerlist('bl_dimp_summary', 'online_mean_cd_thresholded_bl_30', range(5)) + \
               trackerlist('bl_dimp_summary', 'online_mean_cd_thresholded_bl_10', range(5)) + \
               trackerlist('bl_dimp_summary', 'online_mean_cd_thresholded_bl_1', range(5)) + \
               trackerlist('bl_dimp_summary', 'online_mean_cd_thresholded_bl_100', range(5)) + \
               trackerlist('bl_dimp_summary', 'online_mean_cd_thresholded_bl_5', range(5)) + \
               trackerlist('bl_dimp_summary', 'online_mean_cd_thresholded_bl_15', range(5)) + \
               trackerlist('bl_dimp_summary', 'online_rand_bl_5', range(5)) + \
               trackerlist('bl_dimp_summary', 'online_rand_bl_15', range(5))
    dataset = get_dataset('fish')

    return trackers, dataset

def bl_always_query_fish_test():
    trackers = trackerlist('bl_dimp_summary', 'online_xs_cd_bl_1', range(5)) + \
               trackerlist('bl_dimp_summary', 'online_xs_cd_bl_10', range(5)) + \
               trackerlist('bl_dimp_summary', 'online_xs_cd_bl_30', range(5)) + \
               trackerlist('bl_dimp_summary', 'online_xs_cd_bl_60', range(5)) + \
               trackerlist('bl_dimp_summary', 'online_xs_cd_bl_5', range(5)) + \
               trackerlist('bl_dimp_summary', 'online_xs_cd_bl_15', range(5)) + \
               trackerlist('bl_dimp_summary', 'online_xs_cd_bl_100', range(5))
    dataset = get_dataset('fish')

    return trackers, dataset

def dimp_fish_test():
    trackers = trackerlist('dimp_original', 'super_dimp_35', range(1)) + \
        trackerlist('dimp_original', 'super_dimp_35_50opt', range(1))
    dataset = get_dataset('fish')

    return trackers, dataset

def vot_summary_size_test():
    trackers = trackerlist('dimp_summary', 'super_dimp_5', range(1)) + \
               trackerlist('dimp_summary', 'super_dimp_10', range(1)) + \
               trackerlist('dimp_summary', 'super_dimp_15', range(1)) + \
               trackerlist('dimp_summary', 'super_dimp_20', range(1)) + \
               trackerlist('dimp_summary', 'super_dimp_25', range(1)) + \
               trackerlist('dimp_summary', 'super_dimp_30', range(1))

    dataset = get_dataset('vot_2020', 'vot')

    return trackers, dataset 
