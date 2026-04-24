#!/usr/bin/env python
"""
fetch_mar.py
Written by Tyler Sutterley (04/2026)

Syncs MAR regional climate outputs for a given url
    ftp://ftp.climato.be/fettweis
    http://ftp.climato.be/fettweis

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

import os
import logging
import pathlib
import argparse
import traceback
import multiprocessing
import FirnCorr.utilities

# default data directory for SMB and firn models
_default_directory = FirnCorr.utilities.get_cache_path()


# PURPOSE: sync local MAR files for a given URL
def fetch_mar(
    URL,
    directory: str | pathlib.Path | None = _default_directory,
    years: list[int] | None = None,
    timeout: int | None = None,
    processes: int = 0,
    clobber: bool = False,
    mode: int = 0o775,
):
    """
    Syncs MAR regional climate outputs for a given url

    Parameters
    ----------
    URL: object
        URL from :py:class:`FirnCorr.utilities.URL`
    directory: str or pathlib.Path
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

    # directory setup
    directory = pathlib.Path(directory).expanduser().absolute()
    # check if local directory exists and recursively create if not
    local_directory = directory.joinpath("MAR")
    local_directory.mkdir(exist_ok=True, parents=True, mode=mode)

    # regular expression for finding years in file names
    R1 = r"\d+" if not years else r"|".join(map(str, years))
    # find files and reduce to years of interest if specified
    if URL.scheme.startswith("ftp"):
        # find files on ftp server
        remote_files, remote_times = FirnCorr.utilities.ftp_list(
            [URL.netloc, URL.path],
            timeout=timeout,
            basename=True,
            pattern=R1,
            sort=True,
        )
    elif URL.scheme.startswith("http"):
        # find files on http server
        remote_files, remote_times = FirnCorr.utilities.mar_list(
            URL.geturl(),
            timeout=timeout,
            pattern=R1,
            sort=True,
        )

    # sync each data file (in parallel if processes > 0)
    if processes > 0:
        # sync each data file
        out = []
        # set multiprocessing start method
        ctx = multiprocessing.get_context("fork")
        # sync in parallel with multiprocessing Pool
        pool = ctx.Pool(processes=processes)
        # keyword arguments for multiprocessing download function
        kwds = dict(timeout=timeout, clobber=clobber, mode=mode)
        # download remote MAR files to local directory
        for colname, collastmod in zip(remote_files, remote_times):
            remote_path = URL.joinpath(colname)
            local_file = local_directory.joinpath(colname)
            out.append(
                pool.apply_async(
                    _multiprocess,
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
    else:
        # sync each data file in series
        kwds = dict(timeout=timeout, clobber=clobber, mode=mode)
        for colname, collastmod in zip(remote_files, remote_times):
            remote_path = URL.joinpath(colname)
            local_file = local_directory.joinpath(colname)
            output = _download(remote_path, collastmod, local_file, **kwds)
            logging.info(output) if output else None


# PURPOSE: wrapper for running the sync program in multiprocessing mode
def _multiprocess(*args, **kwds):
    """
    Wrapper for running the sync program in ``multiprocessing`` mode
    """
    try:
        output = _download(*args, **kwds)
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
def _download(
    URL,
    mtime: int | float,
    local: str | pathlib.Path,
    **kwargs,
):
    """
    Pull file from a remote host

    Parameters
    ----------
    URL: object
        URL from :py:class:`FirnCorr.utilities.URL`
    mtime: float
        Last modification time of the remote file in seconds since the epoch
    local: str or pathlib.Path
        Path to local file to be synced
    kwargs: dict
        Additional keyword arguments for syncing files
    """
    # check if local version of file exists
    if kwargs["clobber"]:
        why = "overwrite"
    elif not local.exists():
        why = "new"
    elif local.exists() and _newer(mtime, local.stat().st_mtime):
        return
    else:
        why = "old"
    # if file does not exist locally, is to be overwritten, or clobber is set
    # output string for printing files transferred
    output = f"\n\tremote={URL} -->\n\tlocal={local}\n\treason={why}"
    # copy remote file contents to local file
    URL.get(
        timeout=kwargs["timeout"],
        local=local,
        hash=FirnCorr.utilities.get_hash(local),
        label=output,
        mode=kwargs["mode"],
    )
    # keep remote modification time of file and local access time
    os.utime(local, (local.stat().st_atime, mtime))
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
        description="""Syncs MAR regional climate outputs
            """
    )
    parser.add_argument("url", type=str, help="MAR url")
    # working data directory
    parser.add_argument(
        "--directory",
        "-D",
        type=pathlib.Path,
        default=_default_directory,
        help="Working data directory",
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

    # create and parse URL object
    URL = FirnCorr.utilities.URL(args.url)
    # parameters for connection check
    if URL.scheme.startswith("ftp"):
        check_connection = FirnCorr.utilities.check_ftp_connection
        netloc = URL.netloc
    elif URL.scheme.startswith("http"):
        check_connection = FirnCorr.utilities.check_connection
        netloc = URL.parents[-1].geturl()
    # check internet connection
    if check_connection(netloc):
        # run program for URL
        fetch_mar(
            URL,
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
