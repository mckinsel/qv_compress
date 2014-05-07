import h5py
import numpy
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


def write_VQs_to_cmph5(cmph5_file, cmph5_chunk, chunk_indices):

    aln_group = cmph5_file[cmph5_chunk.aln_group_path]
    if not aln_group.get("VQ"):
        full_aln_group_len = len(aln_group['AlnArray'])
        aln_group.create_dataset("VQ", (full_aln_group_len,), dtype='uint8',
                                 chunks=True)
    aln_group["VQ"][cmph5_chunk.aln_group_start:cmph5_chunk.aln_group_end] =\
        chunk_indices


def add_vq_track(cmph5_filename, code_book_filename):

    header_line = open(code_book_filename).readline().strip()
    code_book_features = header_line[2:].split(',')
    raw_codes = numpy.loadtxt(code_book_filename, delimiter=',')

    cmph5_file = h5py.File(cmph5_filename, 'r+')

    for cmph5_chunk in utils.cmph5_chunker(
            cmph5_filename, code_book_features, BUFFER_SIZE):

        chunk_code_indices = get_code_indices(cmph5_chunk.data, raw_codes,
                                              code_book_features)
        write_VQs_to_cmph5(cmph5_file, cmph5_chunk, chunk_code_indices)
    cmph5_file.close()
