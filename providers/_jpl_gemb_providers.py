"""
_jpl_gemb_providers.py (04/2026)
Create GEMB providers for FirnCorr database
"""

import re
import json
import inspect
import pathlib
import posixpath
import argparse
import FirnCorr.utilities

# current file path
filename = inspect.getframeinfo(inspect.currentframe()).filename
filepath = pathlib.Path(filename).absolute().parent
# url encoding function
urlencode = FirnCorr.utilities.urlencode
# default ssl context
_default_ssl_context = FirnCorr.utilities._default_ssl_context
# repository API urls
_zenodo_api_url = "https://zenodo.org/api"


# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Create GEMB providers for FirnCorr database"
            """,
        fromfile_prefix_chars="@",
    )
    # command line parameters
    parser.add_argument(
        "--record", "-R", type=str, default="7130968", help="Zenodo record file"
    )
    parser.add_argument(
        "--pretty", "-p", action="store_true", help="Pretty print the json file"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )
    return parser


def main():
    # Read the system arguments listed after the program
    parser = arguments()
    args, _ = parser.parse_known_args()

    # zenodo API host
    HOST = FirnCorr.utilities.URL(_zenodo_api_url)
    records_api = HOST.joinpath("records", args.record)
    # Create and submit request and load JSON response
    records_response = records_api.load(context=_default_ssl_context)
    version = str(records_response["id"])
    # get all versions of the record
    versions_api = HOST.joinpath("records", version, "versions")
    version_response = versions_api.load(context=_default_ssl_context)
    # regular expression pattern for extracting parameters
    regex_pattern = (
        r"GEMB_(Greenland|Antarctica)(_and_Periphery)?_"
        r"FAC_\d{4}_\d{4}_(.*?)mesh_\d+km_(v.*?).nc$"
    )
    # short names for regions
    regions = dict(Antarctica="ais", Greenland="gris")
    # coordinate reference system
    EPSG = {"Greenland": "EPSG:3413", "Antarctica": "EPSG:3031"}

    # output dictionary for model parameters
    output = {}
    variables = ["centered_FAC", "dFAC", "centered_SMB", "accum_SMB"]
    # for each version of the record
    for hit in version_response["hits"]["hits"]:
        # get version of the record
        # find firn model files
        files = [f for f in hit["files"] if re.search(regex_pattern, f["key"])]
        for file in files:
            # search for pattern in filename
            match = re.search(regex_pattern, file["key"])
            # get region
            model_region = match.group(1)
            region = regions[model_region]
            # get model version
            gemb_version = match.group(4).replace("_", ".")
            model_version = f"GEMB-{gemb_version}"
            # build output dictionary for model version and region
            if model_version in output:
                output[model_version][region] = {}
            else:
                output[model_version] = {region: {}}
            # build full path to model files
            FACfile = posixpath.join("GEMB", gemb_version, file["key"])
            SMBfile = FACfile.replace("FAC", "SMB", 1)
            # append to output dictionary
            output[model_version][region]["model_file"] = [FACfile, SMBfile]
            output[model_version][region]["variables"] = variables
            output[model_version][region]["projection"] = EPSG[model_region]
        output[model_version]["format"] = "GEMB"
        output[model_version]["name"] = model_version
        output[model_version]["reference"] = hit["links"]["doi"]
        output[model_version]["version"] = gemb_version

    # writing model parameters to JSON database file
    json_file = filepath.joinpath("GEMB.json")
    print(f"Writing to {json_file}") if args.verbose else None
    with open(json_file, "w") as fid:
        indent = 4 if args.pretty else None
        json.dump(output, fid, indent=indent, sort_keys=True)


if __name__ == "__main__":
    main()
