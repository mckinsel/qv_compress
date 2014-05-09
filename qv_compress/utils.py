"""Some utility methods for reading and modifying QVs in a cmp.h5 file."""
import collections
import h5py
import numpy 

QUIVER_FEATURES = ('DeletionQV',
                   'DeletionTag',
                   'InsertionQV',
                   'MergeQV',
                   'SubstitutionQV')

TAG_ASSIGNMENTS = ((45, 0), # -
                   (65, 1), # A
                   (67, 2), # C
                   (71, 3), # G
                   (78, 4), # N
                   (84, 5)) # T

def spread_tag(ary, tag_index, inverse=False):
    """Replace tag values with values that are equidistant.

    The ASCII values of the various tags are going to be passed to
    a distance-based clustering algorithm, so it will appear that A (65)
    and C (67) are nearby and could perhaps even be merged to B in a
    cluster center. But that's bad, so we have to replace the ASCII
    values wil values that don't have false proximity.
    
    NB!!:
    This isn't quite right. While this help prevent letter merging
    (changing A or C to B), it tends to favor G's because G is in the
    middle. Need to do a little more to address categorical data.

    Args:
        ary: a numpy array
        tag_index: Index of column in ary corresponding to the tag
        inverse: If True, convert back to original ASCII values.
    """

    for k, v in TAG_ASSIGNMENTS:
        if not inverse:
            ary[:, tag_index][ary[:, tag_index] == k] = v
        else:
            ary[:, tag_index][ary[:, tag_index] == v] = k


def remove_deletions_and_skips(ary, sentinel_index):
    """Remove all instances of skip and deletion characters from an array.

    This is helpful when defining clusters. We don't want meaningless values
    affecting the clustering.

    Args:
        ary: training numpy array
        sentinel_index: column of ary that will have the correct deletion and
            skip values. Usually this can be any column.
    """
    skip_indices = numpy.where(ary[:, sentinel_index] == 0)[0]
    del_indices = numpy.where(ary[:, sentinel_index] == 255)[0]
    rows_to_del = numpy.concatenate((skip_indices, del_indices), 0)

    return numpy.delete(ary, rows_to_del, 0)


def fix_mergeqv(ary, merge_index, new_extreme_value):
    """MergeQV sometimes takes the value 100. This isn't a genuine phred
    score; it just means "not a merge". Rescale these extreme values so
    that they don't dominate the clustering.

    Args:
        ary: the numpy array
        merge_index: the column of the ary corresponding to MergeQV
        new_extreme_value: the desired maximum MergeQV

    Returns:
        None: This modified ary in-place
    """

    ary[:, merge_index][ary[:, merge_index] >= new_extreme_value] = 30

def cmph5_chunker(cmph5_filename, feature_list, chunk_size, max_observations=None):
    """Generator function for reading through a cmp.h5 file in chunks
    and producing appropriately shaped arrays for kmeans. Note that the size
    of the returned chunk will be smaller than chunk_size if the chunk
    would extend past the end of an alignment group.

    Args:
        cmph5_filename: path to a cmph5 file
        feature_list: list of features to read from the cmph5 file. Usually
            just utils.QUIVER_FEATURES
        chunk_size: the number of aligned bases to read at a time
        max_observations: stop after reading this many bases. If None, just
            read until the end of the file

    Yields:
        Tuples of (aln_group_path, aln_group_start, aln_group_end, data)
    """

    cmph5_file = h5py.File(cmph5_filename, 'r')
    aln_group_paths = cmph5_file['AlnGroup/Path']
    CmpH5Chunk = collections.namedtuple(
        'CmpH5Chunk', 'aln_group_path aln_group_start aln_group_end data')
    total_rows = 0

    for aln_group_path in aln_group_paths:
        output_data = numpy.zeros(shape=(chunk_size, len(feature_list)))
        aln_group_pos = 0
        aln_group = cmph5_file[aln_group_path]
        aln_group_len = len(aln_group[feature_list[0]])

        while aln_group_pos < aln_group_len:
            num_rows_to_read = min(aln_group_len - aln_group_pos, chunk_size)
            for feature_i in xrange(len(feature_list)):
                feature_name = feature_list[feature_i]
                output_data[:num_rows_to_read, feature_i] = (
                    aln_group[feature_name][aln_group_pos:
                                            aln_group_pos + num_rows_to_read])

            if num_rows_to_read < chunk_size:
                output_data = numpy.resize(output_data, (num_rows_to_read, len(feature_list)))
            yield CmpH5Chunk(aln_group_path, aln_group_pos,
                             aln_group_pos + num_rows_to_read, output_data)

            aln_group_pos += num_rows_to_read
            total_rows += num_rows_to_read

            if max_observations is not None and total_rows >= max_observations:
                raise StopIteration

            output_data = numpy.zeros(shape=(chunk_size, len(feature_list)))
