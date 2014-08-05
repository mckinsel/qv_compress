"""Entry point for qv_compress. Parses arguments, logs, contains main."""
import argparse
import logging
import numpy
import sys

import qv_compress.code_book
import qv_compress.quantize

log = logging.getLogger('main')

def get_parser():
    """Return an ArgumentParser for qv_compress options."""

    desc = "Apply vector quantization to PacBio quality values in a cmp.h5, SAM, or BAM file"
    parser = argparse.ArgumentParser(prog='qv_compress.py', description=desc)
    parser.add_argument("--debug", help="Output detailed log information.",
                        action='store_true')
    subparsers = parser.add_subparsers(dest="cmd")

    parser_build_code_book = subparsers.add_parser(
        "build_code_book",
        help="Create a QV code book from a cmp.h5, SAM, or BAM file.")

    parser_encode = subparsers.add_parser(
        "encode",
        help="Add VQ values to a cmp.h5, SAM, or BAM file.")


    # build_code_book
    parser_build_code_book.add_argument(
        "training_alignments",
        help="Cmp.h5, SAM, or BAM file from which QV clusters will be created")

    parser_build_code_book.add_argument(
        "num_codes",
        type=int,
        help="Number of clusters to create for the code book.")

    parser_build_code_book.add_argument(
        "num_observations",
        help="The number of bases to use when creating cluster centers.",
        type=int,
        default=1000000)

    parser_build_code_book.add_argument(
        "output_csv",
        help="CSV file where code book will be written.",
        default="code_book.csv")
    
    parser_build_code_book.add_argument(
        "--features_to_cluster",
        help=("A comma separated list of features to cluster. Defaults to "
              "the default Quiver quality values:\nDeletionQV,DeletionTag,"
              "InsertionQV,MergeQV,SubstitutionQV"),
        default="DeletionQV,DeletionTag,InsertionQV,MergeQV,SubstitutionQV",
        type=lambda x: x.split(','))

    # encode_cmp_h5
    parser_encode.add_argument(
        "alignment_file",
        help="File to which a VQ information will be added.")

    parser_encode.add_argument(
        "code_book_csv",
        help="Code book created by 'build_code_book'.")
    
    parser_encode.add_argument(
        "--overwrite_qvs",
        action='store_true',
        default=False,
        help=("Overwrite the QV datasets in the file with "
              "values from the code book."))
    
    parser_encode.add_argument(
        "--rle_deltag",
        action="store_true",
        default="False",
        help=("Apply run length encoding to the DeletionTag and add it as an "
              "optional field. Only works with SAM/BAM files."))

    parser_encode.add_argument(
        "--output_filename",
        action="store",
        default=None,
        help=("Name of new file with VQ-encoded quality values. Only used and "
              "required when the aligment_file is a SAM or BAM."))

    return parser

def check_args(args, parser):
    """Perform a few checks of the arguments. If a check fails, it calls
    parser.error
    """

    if args.cmd == 'encode': 
        if args.alignment_file.endswith(".sam") or args.aligment_file.endswith(".bam"):
            if args.output_filename is None:
                parser.error("When encoding a SAM or BAM file, you have to specify a "
                             "--output_filename.")
        else: 
            if args.rle_deltag:
                parser.error("Run length encoding of DeletionTag only works with SAM "
                             "or BAM files.")



def setup_log(alog, file_name=None, level=logging.DEBUG, str_formatter=None):
    """Util function for setting up logging."""

    if file_name is None:
        handler = logging.StreamHandler(sys.stderr)
    else:
        handler = logging.FileHandler(file_name)

    if str_formatter is None:
        str_formatter = ('[%(levelname)s] %(asctime)-15s '
                         '[%(name)s %(funcName)s %(lineno)d] %(message)s')

    formatter = logging.Formatter(str_formatter)
    handler.setFormatter(formatter)
    alog.addHandler(handler)
    alog.setLevel(level)

def main():
    """Entry point."""

    parser = get_parser()
    args = parser.parse_args()
    check_args(args, parser)

    if args.debug:
        setup_log(log, level=logging.DEBUG)
    else:
        setup_log(log, level=logging.INFO)

    if args.cmd == 'build_code_book':
        code_book, feature_list = qv_compress.code_book.create_code_book(
            args.training_alignments, args.num_codes,
            args.num_observations, args.features_to_cluster)

        numpy.savetxt(args.output_csv, code_book, header=','.join(feature_list),
                      fmt='%2.0f', delimiter=',')
    elif args.cmd == 'encode':
        qv_compress.quantize.add_vqs_to_file(
            args.alignment_file, args.code_book_csv, args.overwrite_qvs,
            args.rle_deltag, args.output_filename)
