"""
fetch_gemb.py
Written by Tyler Sutterley (04/2026)
Downloads Glacier Energy and Mass Balance (GEMB) model outputs

PROGRAM DEPENDENCIES:
    utilities.py: download and management utilities for syncing files

UPDATE HISTORY:
    Written 04/2026
"""

import re
import shutil
import logging
import pathlib
import argparse
import FirnCorr.utilities

# default data directory for SMB and firn models
_default_directory = FirnCorr.utilities.get_cache_path()
# default ssl context
_default_ssl_context = FirnCorr.utilities._default_ssl_context
# repository API urls
_zenodo_api_url = "https://zenodo.org/api"


def fetch_gemb(
    record: str,
    directory: str | pathlib.Path | None = _default_directory,
    timeout: int | None = None,
    clobber: bool = False,
    chunk: int = 16384,
    mode: int = 0o775,
):
    """
    Syncs GEMB model outputs for a given zenodo record

    Parameters
    ----------
    record: str
        Zenodo record number
    directory: str or pathlib.Path
        Working data directory
    timeout: int, default None
        Timeout in seconds for blocking operations
    clobber: bool, default False
        Overwrite existing data
    chunk: int, default 16384
        Chunk size for copying files in bytes
    mode: int, default 0o775
        Permission mode of the local directories and files (number in octal)
    """

    # standard output (terminal output)
    logging.basicConfig(level=logging.INFO)

    # zenodo API host
    HOST = FirnCorr.utilities.URL(_zenodo_api_url)
    records_api = HOST.joinpath("records", record)
    # Create and submit request and load JSON response
    records_response = records_api.load(context=_default_ssl_context)
    version = str(records_response["id"])

    # regular expression pattern for extracting parameters
    regex_pattern = (
        r"GEMB_(Greenland|Antarctica)(_and_Periphery)?_"
        r"(FAC|SMB)_\d{4}_\d{4}_(.*?)(\d+day_)?mesh_\d+km_(v.*?).nc$"
    )
    # get files from latest version of record
    version = str(records_response["id"])
    deposit_api = HOST.joinpath("deposit", "depositions", version, "files")
    # Create and submit request and load JSON response
    deposit_response = deposit_api.load(
        timeout=timeout, context=_default_ssl_context
    )
    # for each file in the JSON response for deposits
    for f in deposit_response:
        # search for pattern in filename
        match = re.search(regex_pattern, f["filename"])
        # check if needing to include algorithm in the hash comparison
        include_algorithm = re.match(r"md5\:", f["checksum"])
        # skip file if pattern is not found
        if not match:
            logging.debug(f"Skipping file: {f['filename']}")
            continue
        # extract parameters from filename
        gemb_version = match.group(6).replace("_", ".")
        # check if local directory exists and recursively create if not
        local_directory = directory.joinpath("GEMB", gemb_version)
        local_directory.mkdir(exist_ok=True, parents=True, mode=mode)
        # full path to output file
        local_file = local_directory.joinpath(f["filename"])
        # check if file already exists by matching MD5 checksums
        original_md5 = FirnCorr.utilities.get_hash(
            local_file, include_algorithm=include_algorithm
        )
        # skip download if checksums match
        if original_md5 == f["checksum"] and not clobber:
            continue
        # download url for remote file
        download = FirnCorr.utilities.URL(f["links"]["download"])
        # output file information
        logging.info(download.urlname)
        # get remote file as a byte-stream
        remote_buffer = download.get(
            timeout=timeout, context=_default_ssl_context
        )
        # verify MD5 checksums
        computed_md5 = FirnCorr.utilities.get_hash(
            remote_buffer, include_algorithm=include_algorithm
        )
        # raise exception if checksums do not match
        if computed_md5 != f["checksum"]:
            raise Exception(f"Checksum mismatch: {download.urlname}")
        # write the file to the local directory
        logging.info(f"\t--> {local_file}")
        with local_file.open(mode="wb") as f:
            shutil.copyfileobj(remote_buffer, f, chunk)
        # change the permissions mode
        local_file.chmod(mode=mode)


# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Downloads Glacier Energy and Mass Balance (GEMB) model outputs
            """,
        fromfile_prefix_chars="@",
    )
    # command line parameters
    # working data directory
    parser.add_argument(
        "--directory",
        "-D",
        type=pathlib.Path,
        default=_default_directory,
        help="Working data directory",
    )
    # zenodo record number
    parser.add_argument(
        "--record",
        "-R",
        type=str,
        default="7130968",
        help="Zenodo record",
    )
    # connection timeout
    parser.add_argument(
        "--timeout",
        "-T",
        type=int,
        default=120,
        help="Timeout in seconds for blocking operations",
    )
    # clobber will overwrite the existing data
    parser.add_argument(
        "--clobber",
        "-C",
        default=False,
        action="store_true",
        help="Overwrite existing data",
    )
    # permissions mode of the local directories and files (number in octal)
    parser.add_argument(
        "--mode",
        "-M",
        type=lambda x: int(x, base=8),
        default=0o775,
        help="Permission mode of directories and files downloaded",
    )
    # return the parser
    return parser


# This is the main part of the program that calls the individual modules
def main():
    # Read the system arguments listed after the program
    parser = arguments()
    args, _ = parser.parse_known_args()

    # run program for record
    fetch_gemb(
        args.record,
        directory=args.directory,
        timeout=args.timeout,
        clobber=args.clobber,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
