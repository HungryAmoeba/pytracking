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
    trackers = trackerlist('dimp_original', 'super_dimp', range(5)) + \
               trackerlist('dimp_summary', 'super_dimp_15', range(5))

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

def fish_test():
    trackers = trackerlist('dimp_original', 'super_dimp', range(5)) + \
               trackerlist('dimp_summary', 'super_dimp_15', range(5))

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
