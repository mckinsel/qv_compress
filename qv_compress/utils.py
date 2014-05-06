import numpy 

QUIVER_FEATURES = ('DeletionQV',
                   'DeletionTag',
                   'InsertionQV',
                   'MergeQV',
                   'SubstitutionQV')

TAG_ASSIGNMENTS = ((45, 0), # -
                   (65, 1), # A
                   (67, 2), # C
                   (71, 3), # T
                   (78, 4), # N
                   (84, 5)) # T

def spread_tag(ary, tag_index, inverse=False):
    """Replace tag values with values that are equidistant.

    The ASCII values of the various tags are going to be passed to
    a distance-based clustering algorithm, so it will appear that A (65)
    and C (67) are nearby and could perhaps even be merged to B in a
    cluster center. But that's bad, so we have to replace the ASCII
    values wil values that don't have false proximity.

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
