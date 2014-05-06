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

    return  training_data

def prepare_training_data(training_array, feature_list):
    """Get the training data ready for clustering. This means fixing a few
    features of the data that will cause problems for clustering and then
    whitening the data.

    Args:
        training_array: the numpy array output by read_from_cmph5
        feature_list: labels for columns of the training array

    Returns:
        None: modifies training_array in-place
    """

    utils.spread_tag(training_array, feature_list.index(['DeletionTag']),
                     inverse=False)
    utils.fix_mergeqv(training_array, feature_list.index('MergeQV'), 40)
    utils.remove_deletions_and_skips(training_array,
                                     feature_list.index("InsertionQV"))

    vq.whiten(training_array)

def create_code_book(cmph5_filename, num_clusters, num_observations,
                     feature_list=utils.QUIVER_FEATURES):

    cmph5_file = h5py.File(cmph5_filename, 'r')
    training_array = read_from_cmph5(cmph5_file, num_observations,
                                     feature_list)
    prepare_training_data(training_array, feature_list)
    code_book, distortion = vq.kmeans(training_array, num_clusters)

    return code_book, feature_list
