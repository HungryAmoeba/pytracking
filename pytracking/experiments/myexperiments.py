from pytracking.evaluation import Tracker, get_dataset, trackerlist


def atom_nfs_uav():
    # Run three runs of ATOM on NFS and UAV datasets
    trackers = trackerlist('atom', 'default', range(3))

    dataset = get_dataset('nfs', 'uav')
    return trackers, dataset


def uav_test():
    # Run DiMP18, ATOM and ECO on the UAV dataset
    trackers = trackerlist('dimp', 'dimp18', range(1)) + \
               trackerlist('atom', 'default', range(1)) + \
               trackerlist('eco', 'default', range(1))

    dataset = get_dataset('uav')
    return trackers, dataset

def dimp_kcenters_test():
    trackers = trackerlist('dimp', 'super_dimp', range(1)) + \
               trackerlist('dimp', 'dimp50', range(1))

    dataset = get_dataset('vot')
    return trackers, dataset

def dimp_kcenters_lasot():
    trackers = trackerlist('dimp', 'super_dimp', range(1)) + \
               trackerlist('dimp', 'dimp50', range(1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def vot_test_k_smaller_30():
    trackers = trackerlist('dimp', 'dimp50_k_30', range(1)) + \
               trackerlist('dimp', 'super_dimp_k_15', range(1))

    dataset = get_dataset('vot')
    return trackers, dataset

def vot_test_comprehensive_size():
    trackers = trackerlist('dimp', 'dimp50_20', range(1)) + \
               trackerlist('dimp', 'dimp50_25', range(1)) + \
               trackerlist('dimp', 'dimp50_30', range(1)) + \
               trackerlist('dimp', 'dimp50_35', range(1)) + \
               trackerlist('dimp', 'dimp50_40', range(1)) + \
               trackerlist('dimp', 'dimp50_45', range(1)) + \
               trackerlist('dimp', 'dimp50', range(1)) + \
               trackerlist('dimp', 'super_dimp_20', range(1)) + \
               trackerlist('dimp', 'super_dimp_25', range(1)) + \
               trackerlist('dimp', 'super_dimp_30', range(1)) + \
               trackerlist('dimp', 'super_dimp_35', range(1)) + \
               trackerlist('dimp', 'super_dimp_40', range(1)) + \
               trackerlist('dimp', 'super_dimp_45', range(1)) + \
               trackerlist('dimp', 'super_dimp', range(1))

    dataset = get_dataset('vot')
    return trackers, dataset

def RLT_DIMP_tests():
    trackers = trackerlist('RLT_dimp_kcenters', 'new', range(1)) + \
               trackerlist('RLT_dimp', 'new', range(1))
    dataset = get_dataset('vot')
    return trackers, dataset

def vot_dimp_partitioned():
    trackers = trackerlist('dimp_partitioned', 'dimp50', range(1)) + \
               trackerlist('dimp_original', 'dimp50', range(1))

    dataset = get_dataset('vot')
    return trackers, dataset

def no_interaction_fish():
    trackers = trackerlist('dimp_original', 'dimp50', range(1)) + \
               trackerlist('dimp_original', 'super_dimp', range(1)) + \
               trackerlist('dimp_partitioned', 'dimp50', range(1)) + \
               trackerlist('dimp_partitioned', 'super_dimp', range(1)) + \
               trackerlist('dimp_summary', 'dimp15', range(1)) + \
               trackerlist('dimp_summary', 'super_dimp_15', range(1))

    dataset = get_dataset('fish')
    return trackers, dataset

def rerun_fish():
    trackers = trackerlist('dimp_summary', 'dimp15', range(1))

    dataset = get_dataset('fish')
    return trackers, dataset

def no_interaction_vot():
    trackers = trackerlist('dimp_original', 'dimp50', range(1)) + \
               trackerlist('dimp_original', 'super_dimp', range(1)) + \
               trackerlist('dimp_partitioned', 'dimp50', range(1)) + \
               trackerlist('dimp_partitioned', 'super_dimp', range(1)) + \
               trackerlist('dimp_summary', 'dimp15', range(1)) + \
               trackerlist('dimp_summary', 'super_dimp_15', range(1))

    dataset = get_dataset('vot')
    return trackers, dataset

def fish_interactive():
    trackers = trackerlist('dimp_summary', 'super_dimp_15_interactive', range(1))

    dataset = get_dataset('fish')

    return trackers, dataset
