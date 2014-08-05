"""code_book contains methods for creating a QV code book from a cmp.h5 file."""
import h5py
import logging
import numpy
import pysam

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

    if "DeletionTag" in feature_list:
        utils.spread_tag(clusterable_array, feature_list.index('DeletionTag'),
                        inverse=False)

    if "MergeQV" in feature_list:
        utils.fix_mergeqv(clusterable_array, feature_list.index('MergeQV'), 40)
    
    # Whiten, unless there's no variance in the column
    if std_dev is None:
        std_dev = numpy.std(clusterable_array, axis=0)

    for i in range(len(std_dev)):
        if std_dev[i] == 0:
            std_dev[i] = 1

    clusterable_array = clusterable_array / std_dev
    return clusterable_array, std_dev

def convert_to_raw(clusterable_array, feature_list, std_dev):
    """Convert an array that's been modified for kmeans back to raw values
    found in a cmp.h5 file.
    """
    
    raw_array = (clusterable_array * std_dev).astype('uint8')

    if "DeletionTag" in feature_list:
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

def read_cmph5(cmph5_filename, feature_list, num_observations):
    """Read training data from a cmph5 file. Returns a numpy array."""

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
    
    return training_array

def read_sam(sam_filename, feature_list, num_observations):

    sam_file = pysam.Samfile(sam_filename)
    num_read_bases = 0

    training_array = numpy.zeros(shape=(num_observations, len(feature_list)))
    
    while num_read_bases < num_observations:
        try:
            sam_record = sam_file.next()
        except StopIteration:
            break
        
        for index, feature_name in enumerate(feature_list):
            values = sam_record.opt(utils.QUIVER_SAM_TAGS[feature_name])
            if feature_name.endswith("Tag"):
                numeric_values = [ord(k) for k in values]
            else:
                numeric_values = [utils.char_to_qv(k) for k in values]

            training_array[
                num_read_bases:min(num_read_bases + len(numeric_values), num_observations),
                index] = numeric_values[:min(len(numeric_values), num_observations - num_read_bases)]
        num_read_bases += len(numeric_values)
    
    if num_read_bases < num_observations:
        log.warning("Only read {n} QV observations, less than the requested {o}"
                    .format(n=num_read_bases, o=num_observations))
        training.array.resize(new_shape=(num_read_bases, len(feature_list)))

    return training_array



def create_code_book(input_filename, num_clusters, num_observations,
                     feature_list=utils.QUIVER_FEATURES):
    """Create a code book from a cmp.h5 file.
    
    Args:
        input_filename: path to the SAM or cmp.h5 file
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
    
    if input_filename.endswith(".cmp.h5"):
        training_array = read_cmph5(input_filename, feature_list, num_observations)
    elif input_filename.endswith(".sam") or input_filename.endswith(".bam"):
        training_array = read_sam(input_filename, feature_list, num_observations)
    else:
        raise RuntimeError, "Input file must be SAM, BAM, or cmp.h5"


    clusterable_array, std_dev = make_data_clusterable(training_array,
                                                       feature_list)
    code_book, distortion = vq.kmeans(clusterable_array, num_clusters)

    raw_code_book = convert_to_raw(code_book, feature_list, std_dev)
    return raw_code_book, feature_list
