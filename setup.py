import os
from setuptools import setup

_REQUIREMENTS_FILE = "REQUIREMENTS.txt"

def _get_local_file(file_name):
    return os.path.join(os.path.dirname(__file__), file_name)


def _get_requirements(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    reqs = [l for l in lines if not l.startswith("#")]
    return reqs

setup(
    name = "qv_compress",
    version="0.1.0",
    packages=['qv_compress'],
    author="Pacific Biosciences",
    author_email="mkinsella@pacificbiosciences.com",
    description="Tools to apply vector quantization to PacBio quality values.",
    entry_points={'console_scripts': ['qv_compress = qv_compress.main:main']},
    license=open("LICENSES.txt").read(),
    zip_safe = False,
    install_requires=_get_requirements(_get_local_file(_REQUIREMENTS_FILE))
)
