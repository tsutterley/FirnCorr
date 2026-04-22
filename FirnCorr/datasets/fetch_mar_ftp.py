#!/usr/bin/env python
"""
fetch_mar_ftp.py
Written by Tyler Sutterley (04/2026)

Syncs MAR regional climate outputs for a given ftp url
    ftp://ftp.climato.be/fettweis

PROGRAM DEPENDENCIES:
    utilities.py: download and management utilities for syncing files

UPDATE HISTORY:
    Updated 04/2026: refactored for new FirnCorr library
    Updated 10/2021: using python logging for handling verbose output
        use utilities module for finding and retrieving files
    Updated 05/2020: added years option to reduce list of files
    Updated 11/2019: added multiprocessing option to run in parallel
    Written 07/2019
"""

from __future__ import print_function

import sys
import os
import time
import logging
import pathlib
import argparse
import traceback
import posixpath
import multiprocessing
import FirnCorr.utilities


# PURPOSE: sync local MAR files with ftp and handle error exceptions
def fetch_mar_ftp(
    parsed_ftp,
    directory: str | pathlib.Path | None = None,
    years: list[int] | None = None,
    timeout: int | None = None,
    processes: int = 0,
    clobber: bool = False,
    mode: int = 0o775,
):
    """
    Syncs MAR regional climate outputs for a given ftp url

    Parameters
    ----------
    parsed_ftp: ParseResult
        Parsed ftp url from :func:`FirnCorr.utilities.urlparse.urlparse`
    directory: str or pathlib.Path, default None
        Working data directory
    years: list, default None
        Years to sync
    timeout: int, default None
        Timeout in seconds for blocking operations
    processes: int, default 0
        Number of processes to use in file downloads (0 for serial)
    clobber: bool, default False
        Overwrite existing data in transfer
    mode: int, default 0o775
        Local permissions mode of directories and files
    """

    # standard output (terminal output)
    logging.basicConfig(level=logging.INFO)

    # check if local directory exists and recursively create if not
    directory = pathlib.Path(directory).expanduser().absolute()
    directory.mkdir(exist_ok=True, parents=True, mode=mode)

    # list directories from ftp
    R1 = r"\d+" if not years else r"|".join(map(str, years))
    # find files and reduce to years of interest if specified
    remote_files, remote_times = FirnCorr.utilities.ftp_list(
        [parsed_ftp.netloc, parsed_ftp.path],
        timeout=timeout,
        basename=True,
        pattern=R1,
        sort=True,
    )

    # sync each data file
    out = []
    # set multiprocessing start method
    ctx = multiprocessing.get_context("fork")
    # sync in parallel with multiprocessing Pool
    pool = ctx.Pool(processes=processes)
    # download remote MAR files to local directory
    for colname, collastmod in zip(remote_files, remote_times):
        remote_path = [parsed_ftp.netloc, parsed_ftp.path, colname]
        local_file = directory.joinpath(colname)
        kwds = dict(timeout=timeout, clobber=clobber, mode=mode)
        out.append(
            pool.apply_async(
                _multiprocess_download,
                args=(remote_path, collastmod, local_file),
                kwds=kwds,
            )
        )
    # start multiprocessing jobs
    # close the pool
    # prevents more tasks from being submitted to the pool
    pool.close()
    # exit the completed processes
    pool.join()
    # print the output string
    for output in out:
        temp = output.get()
        logging.info(temp) if temp else None


# PURPOSE: wrapper for running the sync program in multiprocessing mode
def _multiprocess_download(*args, **kwds):
    """
    Wrapper for running the sync program in ``multiprocessing`` mode
    """
    try:
        output = _ftp_download(*args, **kwds)
    except Exception as exc:
        # if there has been an error exception
        # print the type, value, and stack trace of the
        # current exception being handled
        logging.critical(f"process id {os.getpid():d} failed")
        logging.error(traceback.format_exc())
    else:
        return output


# PURPOSE: pull file from a remote host checking if file exists locally
# and if the remote file is newer than the local file
def _ftp_download(
    remote: list,
    mtime: int | float,
    local: str | pathlib.Path,
    **kwargs,
):
    """
    Pull file from a ftp serve

    Parameters
    ----------
    remote: list
        Remote path components to file on ftp server
    mtime: float
        Last modification time of the remote file in seconds since the epoch
    local: str or pathlib.Path
        Path to local file to be synced
    kwargs: dict
        Additional keyword arguments for syncing files
    """
    # construct the full url to the remote file
    url = posixpath.join(*remote)
    # check if local version of file exists
    if kwargs["clobber"]:
        why = " (overwrite)"
    elif not local.exists():
        why = " (new)"
    elif local.exists() and _newer(mtime, local.stat().st_mtime):
        return
    else:
        why = " (old)"
    # if file does not exist locally, is to be overwritten, or clobber is set
    # output string for printing files transferred
    output = f"{url} -->\n\t{local}{why}\n"
    # copy remote file contents to local file
    FirnCorr.utilities.from_ftp(
        remote,
        timeout=kwargs["timeout"],
        local=local,
        hash=FirnCorr.utilities.get_hash(local),
    )
    # keep remote modification time of file and local access time
    os.utime(local, (local.stat().st_atime, mtime))
    local.chmod(mode=kwargs["mode"])
    # return the output string
    return output


# PURPOSE: compare the modification time of two files
def _newer(t1: int, t2: int) -> bool:
    """
    Compare the modification time of two files

    Parameters
    ----------
    t1: int
        Modification time of first file
    t2: int
        Modification time of second file
    """
    return FirnCorr.utilities.even(t1) <= FirnCorr.utilities.even(t2)


# PURPOSE: create arguments parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Syncs MAR regional climate outputs for a given ftp url
            """
    )
    parser.add_argument("url", type=str, help="MAR ftp url")
    # working data directory
    parser.add_argument(
        "--directory", "-D", type=pathlib.Path, help="Working data directory"
    )
    # years of data to sync
    parser.add_argument(
        "--year", "-Y", type=int, nargs="+", help="Years to sync"
    )
    # run sync in series if processes is 0
    parser.add_argument(
        "--np",
        "-P",
        metavar="PROCESSES",
        type=int,
        default=1,
        help="Number of processes to use in file downloads",
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

    # check internet connection
    if FirnCorr.utilities.check_ftp_connection("ftp.climato.be"):
        # run program for parsed ftp
        parsed_ftp = FirnCorr.utilities.urlparse.urlparse(args.url)
        fetch_mar_ftp(
            parsed_ftp,
            directory=args.directory,
            years=args.year,
            timeout=args.timeout,
            processes=args.np,
            clobber=args.clobber,
            mode=args.mode,
        )


# run main program
if __name__ == "__main__":
    main()
