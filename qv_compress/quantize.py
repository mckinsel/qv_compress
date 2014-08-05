import h5py
import itertools
import json
import numpy
import pysam
from scipy.cluster import vq

from qv_compress import code_book, utils

BUFFER_SIZE = 500000


def get_code_indices(chunk_data, raw_codes, feature_list):
    """Get nearest code index for each row in chunk_data."""

    clusterable_data, std_dev = code_book.make_data_clusterable(
        chunk_data, feature_list, std_dev=None, remove_skips=False)

    whitened_codes, std_dev = code_book.make_data_clusterable(
        raw_codes, feature_list, std_dev=std_dev, remove_skips=False)

    code_indices, distortions = vq.vq(clusterable_data, whitened_codes)
    return code_indices


def write_vqs_to_cmph5(cmph5_file, cmph5_chunk, chunk_indices):

    aln_group = cmph5_file[cmph5_chunk.aln_group_path]
    if not aln_group.get("VQ"):
        full_aln_group_len = len(aln_group['AlnArray'])
        aln_group.create_dataset("VQ", (full_aln_group_len,), dtype='uint8',
                                 chunks=True)
    aln_group["VQ"][cmph5_chunk.aln_group_start:cmph5_chunk.aln_group_end] =\
        chunk_indices

def overwrite_qvs_cmph5_chunk(cmph5_file, chunk, code_indices, code_book,
                              feature_list):
    """Overwrite the quality values in a cmp.h5 file with cluster centers from a
    code book.
    """
    aln_group = cmph5_file[chunk.aln_group_path]
    
    full_feature_array = code_book[code_indices]

    for feature_i in xrange(len(feature_list)):
        feature_name = feature_list[feature_i]
        aln_group[feature_name][chunk.aln_group_start:chunk.aln_group_end] =\
            full_feature_array[...,feature_i]

def read_code_book(code_book_filename):
    """Reads the feature names and cluster values from code book produced
    by create_code_book.

    Returns:
        raw_codes: a numpy.array of the code values
        code_book_features: a list of the names of features associated with
                            columns of raw_codes
    """
    header_line = open(code_book_filename).readline().strip()
    code_book_features = header_line[2:].split(',')
    raw_codes = numpy.loadtxt(code_book_filename, delimiter=',', ndmin=2)
    
    return raw_codes, code_book_features

def run_length_encode(tags_to_encode):
    """Perform run length encoding.

    Args:
        tags_to_encode: A string, usually of DeletionTags.

    Returns:
        the RLE string. For example, if the input was NNNNNCNNTTC,
            the function would return 5N1C2N2T1C
    """

    rle_list = [(len(list(g)), str(k))
                for k,g in itertools.groupby(tags_to_encode)]
    rle_string = ''.join([(str(n) if n > 1 else '') + str(c) for n, c in rle_list])
    return rle_string

def add_vqs_to_file(filename, code_book_filename, overwrite_qvs=False,
                    rle_deltag=False, output_filename=None):
    """Add VQ values to a cmp.h5, SAM, or BAM file. 
    
    If the file is a cmp.h5, write a dataset named VQ to each align
    group that represents the index in the code book. Optionally, overwrite
    all the QVs with values from the code book.

    If the file is a SAM or BAM, write code book indices to the QUAL field
    and put the code book in the header.

    Args:
        filename: path to the cmp.h5, SAM, or BAM file
        code_book_filename: path to the code book produced by build_code_book
        overwrite_qvs: if True, overwrite QVs with values from the code book
    """

    raw_codes, code_book_features = read_code_book(code_book_filename)

    if filename.endswith(".cmp.h5"):

        cmph5_file = h5py.File(filename, 'r+')

        for cmph5_chunk in utils.cmph5_chunker(
                filename, code_book_features, BUFFER_SIZE):

            chunk_code_indices = get_code_indices(cmph5_chunk.data, raw_codes,
                                                  code_book_features)
            write_vqs_to_cmph5(cmph5_file, cmph5_chunk, chunk_code_indices)
            if overwrite_qvs:
                overwrite_qvs_cmph5_chunk(cmph5_file, cmph5_chunk,
                                        chunk_code_indices, raw_codes,
                                        code_book_features)
        cmph5_file.close()

    elif filename.endswith(".sam") or filename.endswith(".bam"):

        in_sam_file = pysam.Samfile(filename)

        header = in_sam_file.header
        if 'CO' not in header:
            header['CO'] = []

        header['CO'].append(json.dumps(raw_codes.tolist()))
        header['CO'].append(json.dumps(code_book_features))

        out_sam_file = pysam.Samfile(output_filename, 'wb', header=header)
        
        encoded_tags = [utils.QUIVER_SAM_TAGS[k] for k in code_book_features]
        if rle_deltag:
            encoded_tags.append(utils.QUIVER_SAM_TAGS["DeletionTag"])

        for record in in_sam_file:
            record_data = numpy.zeros(shape=(len(record.qual), len(code_book_features)))
            for i, feature_name in enumerate(code_book_features):
                record_data[:, i] = [utils.char_to_qv(k) for k in 
                                     record.opt(utils.QUIVER_SAM_TAGS[feature_name])]

            chunk_code_indices = get_code_indices(record_data, raw_codes,
                                                  code_book_features)

            qual_string = ''.join([utils.qv_to_char(k) for k in chunk_code_indices])
            
            if rle_deltag:
                deltag = run_length_encode(record.opt(utils.QUIVER_SAM_TAGS['DeletionTag']))
                record.tags += [('dr', deltag)]
            
            record.tags = [k for k in record.tags if k[0] not in encoded_tags]

            record.qual = qual_string
            out_sam_file.write(record)

        in_sam_file.close()
        out_sam_file.close()
