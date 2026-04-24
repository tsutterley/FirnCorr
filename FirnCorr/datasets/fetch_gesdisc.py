#!/usr/bin/env python
"""
fetch_gesdisc.py
Written by Tyler Sutterley (04/2026)

Syncs MERRA-2 surface mass balance (SMB) related products from the Goddard
    Earth Sciences Data and Information Server Center (GES DISC)
    https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/
    https://wiki.earthdata.nasa.gov/display/EL/How+To+Access+Data+With+Python

Register with NASA Earthdata Login system:
    https://urs.earthdata.nasa.gov

Add "NASA GESDISC DATA ARCHIVE" to Earthdata Applications:
    https://urs.earthdata.nasa.gov/approve_app?client_id=e2WVk8Pw6weeLUKZYOxvTQ

tavgM_2d_int (Vertically Integrated Diagnostics) collection:
    PRECCU (convective rain)
    PRECLS (large-scale rain)
    PRECSN (snow)
    and EVAP (evaporation)
tavgM_2d_glc (Land Ice Surface Diagnostics) collection:
    RUNOFF (runoff over glaciated land)

CALLING SEQUENCE:
    python gesdisc_merra_sync.py --user <username>
    where <username> is your NASA Earthdata username

COMMAND LINE OPTIONS:
    --help: list the command line options
    -U X, --user X: username for NASA Earthdata Login
    -W X, --password X: password for NASA Earthdata Login
    -N X, --netrc X: path to .netrc file for authentication
    -D X, --directory X: working data directory
    -v X, --version X: MERRA-2 version
    -Y X, --year X: years to sync
    -e X, --endpoint X: CMR url endpoint type
    -t X, --timeout X: Timeout in seconds for blocking operations
    -C, --clobber: Overwrite existing data in transfer
    -M X, --mode X: Local permissions mode of the files created

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    dateutil: powerful extensions to datetime
        https://dateutil.readthedocs.io/en/stable/
    lxml: Pythonic XML and HTML processing library using libxml2/libxslt
        https://lxml.de/
        https://github.com/lxml/lxml
    future: Compatibility layer between Python 2 and Python 3
        https://python-future.org/

PROGRAM DEPENDENCIES:
    utilities.py: download and management utilities for syncing files

UPDATE HISTORY:
    Updated 04/2026: refactored for new FirnCorr library
    Updated 05/2023: use pathlib to define and operate on paths
    Updated 06/2022: use CMR queries to find reanalysis granules
    Updated 05/2022: use argparse descriptions within sphinx documentation
    Updated 04/2022: lower case keyword arguments to output spatial
    Updated 10/2021: using python logging for handling verbose output
    Updated 06/2021: new last modified date format on GESDISC servers
    Updated 05/2021: added option for connection timeout (in seconds)
        use try/except for retrieving netrc credentials
    Updated 04/2021: set a default netrc file and check access
        default credentials from environmental variables
    Updated 02/2021: add back MERRA-2 invariant parameters sync
    Updated 01/2021: use argparse to set command line parameters
        using utilities program to build opener and list remote files
    Updated 09/2019: added ssl context to urlopen headers
    Updated 06/2018: using python3 compatible octal, input and urllib
    Updated 03/2018: --directory sets base directory similar to other programs
    Updated 08/2017: use raw_input() to enter NASA Earthdata credentials rather
        than exiting with error
    Updated 05/2017: exception if NASA Earthdata credentials weren't entered
        using os.makedirs to recursively create directories
        using getpass to enter server password securely (remove --password)
    Updated 04/2017: using lxml to parse HTML for files and modification dates
        minor changes to check_connection function to parallel other programs
    Written 11/2016
"""

from __future__ import print_function

import os
import time
import shutil
import logging
import pathlib
import argparse
import FirnCorr.utilities

# default data directory for SMB and firn models
_default_directory = FirnCorr.utilities.get_cache_path()


# PURPOSE: download MERRA-2 files from GESDISC
def fetch_gesdisc(
    client,
    directory: str | pathlib.Path | None = _default_directory,
    version: str | None = None,
    years: list | None = None,
    endpoint: str | None = None,
    timeout: int | None = None,
    clobber: bool = False,
    mode: int = 0o775,
):
    """
    Download MERRA-2 files from GESDISC

    Parameters
    ----------
    client: obj
        AWS s3 client for GES DISC
    directory: str or pathlib.Path, default None
        Working data directory
    version: str, default None
        MERRA-2 version
    years: list, default None
        Years of model outputs to sync
    endpoint: str or None, default None
        CMR url endpoint type
    timeout: int, default None
        Timeout in seconds for blocking operations
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
    local_directory = directory.joinpath("MERRA2")
    local_directory.mkdir(exist_ok=True, parents=True, mode=mode)
    # set default dates to download
    if years is None:
        years = range(1980, time.gmtime().tm_year + 1)

    # provider for CMR queries
    provider = FirnCorr.utilities._s3_providers["gesdisc"]
    # query CMR for model MERRA-2 invariant products
    ids, urls, mtimes = FirnCorr.utilities.cmr(
        "M2C0NXASM",
        version=version,
        provider=provider,
        endpoint=endpoint,
        verbose=True,
    )
    # copy files from remote directory comparing modified dates
    for fid, url, mtime in zip(ids, urls, mtimes):
        remote = FirnCorr.utilities.URL(url)
        local = local_directory.joinpath(fid)
        _download(
            remote,
            mtime,
            local,
            client=client,
            timeout=timeout,
            clobber=clobber,
            mode=mode,
        )

    # for each MERRA-2 product to sync
    for shortname in ["M2TMNXINT", "M2TMNXGLC"]:
        product = f"{shortname}.{version}"
        logging.info(f"product={product}")
        # for each year to sync
        for Y in map(str, years):
            # start and end date for query
            start_date = f"{Y}-01-01"
            end_date = f"{Y}-12-31"
            ids, urls, mtimes = FirnCorr.utilities.cmr(
                shortname,
                version=version,
                start_date=start_date,
                end_date=end_date,
                provider=provider,
                endpoint=endpoint,
                verbose=True,
            )
            # copy file from remote directory comparing modified dates
            for fid, url, mtime in zip(ids, urls, mtimes):
                remote = FirnCorr.utilities.URL(url)
                # recursively create local directory for data
                local = local_directory.joinpath(product, Y, fid)
                local.parent.mkdir(mode=mode, parents=True, exist_ok=True)
                _download(
                    remote,
                    mtime,
                    local,
                    client=client,
                    timeout=timeout,
                    clobber=clobber,
                    mode=mode,
                )


# PURPOSE: pull file from a remote host checking if file exists locally
# and if the remote file is newer than the local file
def _download(
    URL,
    mtime: int | float,
    local: str | pathlib.Path,
    chunk: int = 16384,
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
    chunk: int, default 16384
        Chunk size for copying files in bytes
    kwargs: dict
        Additional keyword arguments for syncing files
    """
    # verify local
    local = pathlib.Path(local).expanduser().absolute()
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
    if URL.scheme.startswith("s3"):
        logging.info(output)
        # get object from s3 client and copy to local file
        response = kwargs["client"].get_object(
            Bucket=URL.s3bucket, Key=URL.s3key
        )
        with local.open(mode="wb") as f:
            shutil.copyfileobj(response["Body"], f, chunk)
    else:
        # copy remote file contents to local file
        URL.get(
            context=None,
            timeout=kwargs["timeout"],
            local=local,
            hash=FirnCorr.utilities.get_hash(local),
            chunk=chunk,
            label=output,
        )
    # keep remote modification time of file and local access time
    os.utime(local, (local.stat().st_atime, mtime))
    # change the permissions of the local file
    local.chmod(mode=kwargs["mode"])


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


# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Syncs MERRA-2 surface mass balance (SMB)
            variables from the Goddard Earth Sciences Data and
            Information Server Center (GES DISC)
            """
    )
    # command line parameters
    # NASA Earthdata credentials
    parser.add_argument(
        "--user",
        "-U",
        type=str,
        default=os.environ.get("EARTHDATA_USERNAME"),
        help="Username for NASA Earthdata Login",
    )
    parser.add_argument(
        "--password",
        "-W",
        type=str,
        default=os.environ.get("EARTHDATA_PASSWORD"),
        help="Password for NASA Earthdata Login",
    )
    parser.add_argument(
        "--netrc",
        "-N",
        type=pathlib.Path,
        default=pathlib.Path.home().joinpath(".netrc"),
        help="Path to .netrc file for authentication",
    )
    # working data directory
    parser.add_argument(
        "--directory",
        "-D",
        type=pathlib.Path,
        default=_default_directory,
        help="Working data directory",
    )
    # MERRA-2 version
    parser.add_argument(
        "--version", "-v", type=str, default="5.12.4", help="MERRA-2 version"
    )
    # years to download
    now = time.gmtime()
    parser.add_argument(
        "--year",
        "-Y",
        type=int,
        nargs="+",
        default=range(1980, now.tm_year + 1),
        help="Years of model outputs to sync",
    )
    # CMR endpoint type
    parser.add_argument(
        "--endpoint",
        "-e",
        type=str,
        default="data",
        choices=["s3", "data"],
        help="CMR url endpoint type",
    )
    # connection timeout
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=360,
        help="Timeout in seconds for blocking operations",
    )
    # sync options
    parser.add_argument(
        "--clobber",
        "-C",
        default=False,
        action="store_true",
        help="Overwrite existing data in transfer",
    )
    # permissions mode of the directories and files synced (number in octal)
    parser.add_argument(
        "--mode",
        "-M",
        type=lambda x: int(x, base=8),
        default=0o775,
        help="Permission mode of directories and files synced",
    )
    # return the parser
    return parser


# This is the main part of the program that calls the individual functions
def main():
    # Read the system arguments listed after the program
    parser = arguments()
    args, _ = parser.parse_known_args()

    # NASA Earthdata hostname
    URS = "urs.earthdata.nasa.gov"
    # host for retrieving AWS S3 credentials
    HOST = FirnCorr.utilities._s3_endpoints["gesdisc"]
    # There are a range of exceptions that can be thrown here
    # including HTTPError and URLError.
    if args.endpoint == "s3":
        # build opener for s3 client access
        opener = FirnCorr.utilities.attempt_login(
            URS, username=args.user, password=args.password, netrc=args.netrc
        )
        # Create and submit request to create AWS session
        client = FirnCorr.utilities.s3_client(HOST, args.timeout)
    else:
        # build opener for data client access
        opener = FirnCorr.utilities.attempt_login(
            URS,
            username=args.user,
            password=args.password,
            netrc=args.netrc,
            password_manager=True,
            authorization_header=False,
        )
        client = None

    # retrieve data objects from s3 client or data endpoints
    fetch_gesdisc(
        client,
        directory=args.directory,
        version=args.version,
        years=args.year,
        endpoint=args.endpoint,
        timeout=args.timeout,
        clobber=args.clobber,
        mode=args.mode,
    )


# run main program
if __name__ == "__main__":
    main()
