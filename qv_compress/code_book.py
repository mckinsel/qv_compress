"""code_book contains methods for creating a QV code book from a cmp.h5 file."""
import h5py
import logging
import numpy

from scipy.cluster import vq
from qv_compress import utils

log = logging.getLogger('main')

def make_data_clusterable(training_array, feature_list, std_dev=None, remove_skips=True):
    """Convert data to and from raw and clusterable states. The raw QVs
    have some attributes that will break clustering, so those have to be
    cleaned up before we pass anything to k-means. 

    Args:
        training_array: the numpy array output by read_from_cmph5
        feature_list: labels for columns of the training array
        std_dev: the standard deviation of each columns of training array,
            for whitening. If None, this is calculated from training_array

    Returns:
        clusterable_array: the modified array that can now be clustered
        std_dev: the standard deviations of columns
    """
    
    # Fix the PacBio-specific problems
    if remove_skips:
        clusterable_array = utils.remove_deletions_and_skips(
            training_array, feature_list.index("InsertionQV"))
    else:
        clusterable_array = numpy.array(training_array, copy=True)
    utils.spread_tag(clusterable_array, feature_list.index('DeletionTag'),
                     inverse=False)
    utils.fix_mergeqv(clusterable_array, feature_list.index('MergeQV'), 40)
    
    # Whiten
    if std_dev is None:
        std_dev = numpy.std(clusterable_array, axis=0)
    clusterable_array = clusterable_array / std_dev
    return clusterable_array, std_dev

def convert_to_raw(clusterable_array, feature_list, std_dev):
    """Convert an array that's been modified for kmeans back to raw values
    found in a cmp.h5 file.
    """
    
    raw_array = (clusterable_array * std_dev).astype('uint8')
    utils.spread_tag(raw_array, feature_list.index("DeletionTag"),
                     inverse=True)

    return raw_array

def check_for_features(cmph5_file, feature_list):
    """Check that all required features present in the cmph5_file. Return
    a list of features that are missing.
    """

    aln_group_path = cmph5_file['AlnGroup/Path'][0]
    missing_features = []
    for feature in feature_list:
        if feature not in cmph5_file[aln_group_path].keys():
            missing_features.append(feature)

    return missing_features


def create_code_book(cmph5_filename, num_clusters, num_observations,
                     feature_list=utils.QUIVER_FEATURES):
    """Create a code book from a cmp.h5 file.
    
    Args:
        cmph5_filename: path to the cmp.h5
        num_clusters: the number of codes to create in the code book
        num_observations: the number of bases to use to create the code book
            clusters
        feature_list: the list of features to read from the cmp.h5 to 
            cluster

    Returns:
        code_book: a numpy array of cluster centers. rows are codes, columns are
            features
        feature_list: labels for the columns of the code book
    """

    log.debug("Checking for missing features...")

    cmph5_file = h5py.File(cmph5_filename, 'r')
    missing_features = check_for_features(cmph5_file, feature_list)
    if missing_features:
        raise ValueError("Cmp.h5 file {c} is missing the following quality "
                         "values: {m}"
                         .format(c=cmph5_filename, m=missing_features))
    cmph5_file.close()
    log.debug("All required features present!")

    training_array = numpy.concatenate(
        [k.data for k in utils.cmph5_chunker(
            cmph5_filename, feature_list, num_observations, num_observations)], axis=0)

    clusterable_array, std_dev = make_data_clusterable(training_array,
                                                       feature_list)
    code_book, distortion = vq.kmeans(clusterable_array, num_clusters)

    raw_code_book = convert_to_raw(code_book, feature_list, std_dev)
    return raw_code_book, feature_list
