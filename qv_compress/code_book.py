"""code_book contains methods for creating a QV code book from a cmp.h5 file."""

import h5py
import numpy

from scipy.cluster import vq
from qv_compress import utils

def read_from_cmph5(cmph5_file, num_observations, feature_list):
    """Read desired features from a cmp.h5 file and return them shaped
    appropriately for scipy.cluster.vq.

    Args:
        cmph5_file: An h5py.File object for a cmp.h5 file
        num_observations: the desired number of observations to read
        feature_list: list of feature names to read. This really, really should
            just be QUIVER_FEATURES.

    Returns:
        training_data: a numpy array with shape
            (num_observations x len(feature_list))
    """

    # I know they're not floats, but these observations will need to be
    # centered and whitened before clustering
    training_data = numpy.zeros(shape=(num_observations, len(feature_list)),
                                dtype='float32')

    aln_group_paths = cmph5_file['AlnGroup/Path']
    observations_so_far = 0

    for aln_group_path in aln_group_paths:
        observations_to_read = min(
            num_observations - observations_so_far,
            len(cmph5_file[aln_group_path][feature_list[0]]))

        for feature_i in xrange(len(feature_list)):
            feature = feature_list[feature_i]
            training_data[
                observations_so_far:observations_so_far + observations_to_read,
                feature_i] = (cmph5_file[aln_group_path][feature]
                                        [:observations_to_read])

        observations_so_far += observations_to_read
        if observations_so_far >= num_observations:
            break

    if observations_so_far < num_observations:
        warnings.warn("Only found {n} observations in the cmp.h5 file, less "
                      "than the requested {r}."
                      .format(n=observations_so_far, r=num_observations))
        numpy.resize(training_data, (observations_so_far, len(feature_list)))
    return  training_data

def make_data_clusterable(training_array, feature_list, std_dev=None):
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
    clusterable_array = utils.remove_deletions_and_skips(
        training_array, feature_list.index("InsertionQV"))
    utils.spread_tag(clusterable_array, feature_list.index('DeletionTag'),
                     inverse=False)
    utils.fix_mergeqv(clusterable_array, feature_list.index('MergeQV'), 40)
    
    # Whiten
    std_dev = numpy.std(clusterable_array, axis=0)
    clusterable_array = clusterable_array / std_dev
    return clusterable_array, std_dev

def convert_to_raw(clusterable_array, feature_list, std_dev):
    
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
    """Create a code book."""
    cmph5_file = h5py.File(cmph5_filename, 'r')

    missing_features = check_for_features(cmph5_file, feature_list)
    if missing_features:
        raise ValueError("Cmp.h5 file {c} is missing the following quality "
                         "values: {m}"
                         .format(c=cmph5_filename, m=missing_features))

    training_array = read_from_cmph5(cmph5_file, num_observations,
                                     feature_list)
    clusterable_array, std_dev = make_data_clusterable(training_array,
                                                       feature_list)
    code_book, distortion = vq.kmeans(clusterable_array, num_clusters)
    
    raw_code_book = convert_to_raw(code_book, feature_list, std_dev)
    return raw_code_book, feature_list
