"""Entry point for qv_compress. Parses arguments, logs, contains main."""
import argparse
import logging
import numpy
import sys

import qv_compress.code_book

log = logging.getLogger(__name__)

def get_parser():
    """Return an ArgumentParser for qv_compress options."""

    desc = "Apply vector quantization to PacBio quality values in a cmp.h5"
    parser = argparse.ArgumentParser(prog='qv_compress.py', description=desc)
    parser.add_argument("--debug", help="Output detailed log information.")
    subparsers = parser.add_subparsers(dest="cmd")

    parser_build_code_book = subparsers.add_parser(
        "build_code_book",
        help="Create a QV code book from a cmp.h5 file.")

    parser_add_vq_track = subparsers.add_parser(
        "add_vq_track",
        help=("Add an array of code assignments to an cmp.h5 file from an "
              "existing code book."))

    parser_replace_qvs = subparsers.add_parser(
        "replace_qvs",
        help=("Replace Quiver QVs in a cmp.h5 file with values from a code "
              "book. Note that this is a destructive operation."))

    #build_code_book
    parser_build_code_book.add_argument(
        "training_cmp_h5",
        help="Cmp.h5 file from which QV clusters will be created")

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

    #add_vq_track
    parser_add_vq_track.add_argument(
        "cmp_h5_file",
        help="File to which a VQ track will be added.")

    parser_add_vq_track.add_argument(
        "code_book_csv",
        help="Code book created by 'build_code_book'.")

    #replace_qvs
    parser_replace_qvs.add_argument(
        "cmp_h5_file",
        help="File to which a VQ track will be added.")

    parser_replace_qvs.add_argument(
        "code_book_csv",
        help="Code book created by 'build_code_book'.")

    return parser

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

    if args.debug:
        setup_log(log, level=logging.DEBUG)
    else:
        setup_log(log, level=logging.INFO)

    if args.cmd == 'build_code_book':
        code_book, feature_list = qv_compress.code_book.create_code_book(
            args.training_cmp_h5, args.num_codes,
            args.num_observations)

        numpy.savetxt(args.output_csv, code_book, header=','.join(feature_list),
                     fmt='%2.0f', delimiter=',')
    else:
        print "Soon..."
