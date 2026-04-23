#!/usr/bin/env python
"""
model.py
Written by Tyler Sutterley (04/2026)
Retrieves model parameters for named SMB and firn models

PYTHON DEPENDENCIES:
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/
        https://pyproj4.github.io/pyproj/
    xarray: N-D labeled arrays and datasets in Python
        https://docs.xarray.dev/en/stable/

UPDATE HISTORY:
    Written 04/2026
"""

from __future__ import annotations

import io
import copy
import json
import pyproj
import pathlib
import warnings
import xarray as xr
import FirnCorr.utilities

# suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

__all__ = ["DataBase", "load_database", "model"]

# default working data directory for SMB and firn models
_default_directory = FirnCorr.utilities.get_cache_path()
# default variable attributes
_attributes = dict(
    SMB=dict(long_name="Surface Mass Balance"),
    zsurf=dict(long_name="Total Surface Height Change"),
    zfirn=dict(
        standard_name="firn air content",
        long_name="Snow Height Change due to Compaction",
    ),
    zmelt=dict(long_name="Snow Height Change due to Melt"),
    zsmb=dict(long_name="Snow Height Change due to SMB"),
    zaccum=dict(long_name="Snow Height Change due to Accumulation"),
)


# allow model database to be subscriptable
# and have attribute access
class DataBase:
    """FirnCorr model database and parameters"""

    def __init__(self, d: dict):
        self.__dict__ = d

    def update(self, d: dict):
        """Update the keys of the model database"""
        self.__dict__.update(d)

    def keys(self):
        """Returns the keys of the model database"""
        return self.__dict__.keys()

    def values(self):
        """Returns the values of the model database"""
        return self.__dict__.values()

    def items(self):
        """Returns the items of the model database"""
        return self.__dict__.items()

    def __str__(self):
        """String representation of the ``DataBase`` object"""
        return str(self.__dict__)

    def __repr__(self):
        """Representation of the ``DataBase`` object"""
        return self.__str__()

    def get(self, key, default=None):
        if not hasattr(self, key) or getattr(self, key) is None:
            return default
        else:
            return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)


# PURPOSE: load the JSON database of model files
def load_database(extra_databases: list = []):
    """
    Load the ``JSON`` database of model files

    Parameters
    ----------
    extra_databases: list, default []
        A list of additional databases to load, as either
        ``JSON`` file paths or dictionaries

    Returns
    -------
    parameters: dict
        Database of model parameters
    """
    # path to model database
    database = FirnCorr.utilities.get_data_path(["data", "database.json"])
    # extract JSON data
    with database.open(mode="r", encoding="utf-8") as fid:
        parameters = json.load(fid)
    # verify that extra_databases is iterable
    if isinstance(extra_databases, (str, pathlib.Path, dict)):
        extra_databases = [extra_databases]
    # load any additional databases
    for db in extra_databases:
        # use database parameters directly if a dictionary
        if isinstance(db, dict):
            extra_database = copy.copy(db)
        # otherwise load parameters from JSON file path
        else:
            # verify that extra database file exists
            db = pathlib.Path(db)
            if not db.exists():
                raise FileNotFoundError(db)
            # extract JSON data
            with db.open(mode="r", encoding="utf-8") as fid:
                extra_database = json.load(fid)
        # Add additional models to database
        parameters.update(extra_database)
    return DataBase(parameters)


class model:
    """Retrieves SMB and firn model parameters for named models

    Attributes
    ----------
    compressed: bool
        Model files are gzip compressed
    directory: str, pathlib.Path or None, default None
        Working data directory for SMB and firn models
    extra_databases: list, default []
        Additional databases for model parameters
    verify: bool
        Verify that all model files exist
    """

    def __init__(
        self,
        directory: str | pathlib.Path | None = None,
        **kwargs,
    ):
        # set default keyword arguments
        kwargs.setdefault("compressed", False)
        kwargs.setdefault("verify", True)
        kwargs.setdefault("extra_databases", [])
        # set initial attributes
        self.compressed = copy.copy(kwargs["compressed"])
        # set working data directory
        self.directory = None
        if directory is not None:
            self.directory = pathlib.Path(directory)
        # set any extra databases
        self.extra_databases = copy.copy(kwargs["extra_databases"])
        self.format = None
        self.name = None
        self.region = None
        self.verify = copy.copy(kwargs["verify"])
        self.__parameters__ = {}

    def from_database(
        self,
        m: str,
        group: tuple = ("ais", "gris"),
    ):
        """
        Create a model object from database of known models

        Parameters
        ----------
        m: str
            Model name
        group: tuple, default ("ais", "gris")
            Model region(s) to extract
        """
        # set working data directory if unset
        if self.directory is None:
            self.directory = pathlib.Path(_default_directory)
        # select between known models
        parameters = load_database(extra_databases=self.extra_databases)
        # try to extract parameters for model
        try:
            self.from_dict(parameters[m])
        except (ValueError, KeyError, AttributeError) as exc:
            raise ValueError(f"Unlisted model {m}") from exc
        # verify model types to extract
        if isinstance(group, str):
            group = (group,)
        # verify paths
        for g in group:
            # verify model region is valid
            g = g.lower()
            # skip if model region is unavailable
            if not hasattr(self, g):
                continue
            # validate paths: model constituent files
            self[g].model_file = self.pathfinder(self[g].model_file)
        # return the model parameters
        self.validate_format()
        # set dictionary of parameters
        self.__parameters__ = self.to_dict(serialize=True)
        return self

    def from_file(
        self,
        definition_file: str | pathlib.Path | io.IOBase,
        **kwargs,
    ):
        """
        Create a model object from an input definition file

        Parameters
        ----------
        definition_file: str, pathlib.Path or io.IOBase
            Model definition file for creating model object
        """
        # load and parse definition file
        if isinstance(definition_file, io.IOBase):
            self._parse_file(definition_file)
        elif isinstance(definition_file, (str, pathlib.Path)):
            definition_file = pathlib.Path(definition_file)
            with definition_file.open(mode="r", encoding="utf8") as fid:
                self._parse_file(fid)
        # set dictionary of parameters
        self.__parameters__ = self.to_dict(serialize=True)
        # return the model object
        return self

    def from_dict(self, d: dict):
        """
        Create a model object from a dictionary of parameters

        Parameters
        ----------
        d: dict
            Model object parameters
        """
        # copy model parameters
        self.__parameters__ = copy.copy(d)
        for key, val in d.items():
            if isinstance(val, dict) and key not in ("projection",):
                setattr(self, key, DataBase(val))
            else:
                setattr(self, key, copy.copy(val))
        # return the model parameters
        return self

    def to_dict(self, **kwargs):
        """
        Create a dictionary from a model object

        Parameters
        ----------
        fields: list, default all
            List of model attributes to output
        serialize: bool, default False
            Serialize dictionary for ``JSON`` output
        """
        # default fields
        fields = ["name", "format", "projection", "reference"]
        # set default keyword arguments
        kwargs.setdefault("fields", fields + ["ais", "gris"])
        kwargs.setdefault("serialize", False)
        # output dictionary
        d = {}
        # for each field
        for key in kwargs["fields"]:
            if hasattr(self, key) and getattr(self, key) is not None:
                d[key] = getattr(self, key)
        # serialize dictionary for JSON output
        if kwargs["serialize"]:
            d = self.serialize(d)
        # return the model dictionary
        return d

    @property
    def gzip(self) -> str:
        """Returns suffix for ``gzip`` compression"""
        return ".gz" if self.compressed else ""

    @property
    def engine(self) -> str:
        """
        Returns the read-write engine for the model
        """
        part1, _, part2 = self.format.partition("-")
        if "-" in self.format:
            return part1
        else:
            return self.format

    @property
    def file_format(self) -> str:
        """
        Returns the file format for the model
        """
        part1, _, part2 = self.format.partition("-")
        if "-" in self.format:
            return part2
        else:
            return self.format

    @property
    def crs(self):
        """Coordinate reference system of the model"""
        # default is EPSG:4326 (WGS84)
        CRS = self.get("projection", 4326)
        return pyproj.CRS.from_user_input(CRS)

    def pathfinder(
        self,
        model_file: str | pathlib.Path | list,
    ):
        """
        Completes file paths and appends ``gzip`` suffix

        Parameters
        ----------
        model_file: str, pathlib.Path or list
            Model file(s) to complete
        """
        # set working data directory if unset
        if self.directory is None:
            self.directory = pathlib.Path(_default_directory)
        # complete model file paths
        if isinstance(model_file, list):
            output_file = [self.pathfinder(f) for f in model_file]
            valid = all([f.exists() for f in output_file])
        elif isinstance(model_file, str):
            output_file = self.directory.joinpath(
                "".join([model_file, self.gzip])
            )
            valid = output_file.exists()
        # check that (all) output files exist
        if self.verify and not valid and not self.compressed:
            # try seeing if there are compressed files
            self.compressed = True
            output_file = self.pathfinder(model_file)
        elif self.verify and not valid:
            raise FileNotFoundError(output_file)
        # return the complete output path
        return output_file

    def _parse_file(self, fid: io.IOBase):
        """
        Load and parse a model definition file

        Parameters
        ----------
        fid: io.IOBase
            Open definition file object
        """
        # attempt to read and parse a JSON file
        try:
            self._parse_json(fid)
        except json.decoder.JSONDecodeError as exc:
            pass
        else:
            return self
        # raise an exception
        raise IOError("Cannot load model definition file")

    def _parse_json(self, fid: io.IOBase):
        """
        Load and parse ``JSON`` definition file

        Parameters
        ----------
        fid: io.IOBase
            Open definition file object
        """
        # load JSON file
        parameters = json.load(fid)
        # convert from dictionary to model variable
        temp = self.from_dict(parameters)
        # verify model name and format
        assert temp.name
        temp.validate_format()
        # return the model parameters
        return temp

    def validate_format(self):
        """Asserts that the model format is a known type"""
        # assert that model is a known format
        known_formats = []
        known_formats.append("GSFC-fdm")
        known_formats.append("MAR")
        known_formats.append("RACMO-ascii")
        known_formats.append("RACMO-netcdf")
        known_formats.append("RACMO-downscaled")
        assert self.format in known_formats

    def serialize(self, d: dict):
        """
        Encodes dictionary to be ``JSON`` serializable

        Parameters
        ----------
        d: dict
            Parameters to serialize
        """
        # iterate over keys
        for key, val in d.items():
            val = copy.copy(d[key])
            if isinstance(val, pathlib.Path):
                d[key] = str(val)
            elif isinstance(val, (list, tuple)) and isinstance(
                val[0], pathlib.Path
            ):
                d[key] = [str(v) for v in val]
            elif isinstance(val, dict):
                d[key] = self.serialize(val)
            elif isinstance(val, DataBase):
                d[key] = self.serialize(val.__dict__)
        # return the model dictionary
        return d

    def open_dataset(self, **kwargs):
        """
        Open model files as an xarray Dataset

        Parameters
        ----------
        kwargs: dict
            Additional keyword arguments for opening model files

        Returns
        -------
        ds: xarray.Dataset
            SMB and firn model data
        """
        # import SMB and firn model functions
        from FirnCorr.io import GSFCfdm, MAR, RACMO

        # set default keyword arguments
        kwargs.setdefault("use_default_units", False)
        kwargs.setdefault("compressed", self.compressed)
        # model group to extract
        group = kwargs["group"].lower()
        assert group in ("ais", "gris"), f"Invalid model group {group}"
        # extract model file
        model_file = self[group].get("model_file")
        # extract default model variables
        kwargs.setdefault("variable", self[group].get("variables"))
        # extract dataset from model file(s)
        if self.engine == "GSFC":
            # open GSFCfdm file(s) as xarray Dataset
            ds = GSFCfdm.open_dataset(model_file, **kwargs)
        elif self.engine == "MAR":
            # open MAR file(s) as xarray Dataset
            ds = MAR.open_mfdataset(model_file, **kwargs)
            # calculate derived fields (if available)
            if all(v in kwargs["variables"] for v in ["ZN6", "ZN4"]):
                ds["zsmb"] = ds["zsurf"] - ds["zfirn"]
                ds["zsmb"].attrs.update(_attributes["zsmb"])
                ds["zsmb"].attrs["group"] = ["ZN6", "ZN4"]
            if all(v in kwargs["variables"] for v in ["ZN6", "ZN4", "ZN5"]):
                ds["zaccum"] = ds["zsurf"] - ds["zfirn"] - ds["zmelt"]
                ds["zaccum"].attrs.update(_attributes["zaccum"])
                ds["zaccum"].attrs["group"] = ["ZN6", "ZN4", "ZN5"]
        elif self.engine == "RACMO":
            # open RACMO file(s) as xarray Dataset
            ds = RACMO.open_mfdataset(
                model_file,
                format=self.file_format,
                **kwargs,
            )
        # add attributes
        ds.attrs["source"] = self.name
        # update projection attributes
        self.projection = ds.attrs.get("crs", self.crs.to_dict())
        # convert to default units
        if kwargs["use_default_units"]:
            ds = ds.smb.to_default_units()
        # return xarray dataset
        return ds

    def open_datatree(
        self,
        group: tuple = ("ais", "gris"),
        **kwargs,
    ):
        """
        Open model files as an xarray DataTree

        Parameters
        ----------
        group: tuple, default ('ais', 'gris')
            List of model types to extract
        kwargs: dict
            Additional keyword arguments for opening model files

        Returns
        -------
        dtree: xr.DataTree
            SMB and firn model data
        """
        # output dictionary of xarray Datasets
        ds = {}
        # try to read model files
        for g in group:
            # skip if model group is unavailable
            if not hasattr(self, g.lower()):
                continue
            # open xarray Dataset
            ds[g] = self.open_dataset(group=g, **kwargs)
        # create xarray DataTree from dictionary
        dtree = xr.DataTree.from_dict(ds)
        # return the model xarray DataTree
        return dtree

    def __str__(self):
        """String representation of the ``io.model`` object"""
        properties = ["FirnCorr.io.model"]
        properties.append(f"    name: {self.name}")
        properties.append(f"    area: {self.region}")
        return "\n".join(properties)

    def __repr__(self):
        """Representation of the ``io.model`` object"""
        return self.__str__()

    def _repr_html_(self):
        """HTML representation of the ``io.model`` object"""
        header = "FirnCorr.io.model"
        header_components = [f"<div class='xr-obj-type'>{header}</div>"]
        sections = []
        data_vars = self.__variables__.copy()
        parameters = {
            k: v for k, v in self.__parameters__.items() if k not in data_vars
        }
        sections.append(xr.core.formatting_html.attr_section(parameters))
        for v in data_vars:
            sections.append(
                xr.core.formatting_html._mapping_section(
                    mapping=self.__parameters__[v],
                    name=f"{v}-Attributes",
                    details_func=xr.core.formatting_html.summarize_attrs,
                    max_items_collapse=0,
                    expand_option_name="display_expand_attrs",
                )
            )
        return xr.core.formatting_html._obj_repr(
            self, header_components, sections
        )

    @property
    def __variables__(self):
        """List of model variables"""
        return [k for k in ("ais", "gris") if k in self.__parameters__]

    def get(self, key, default=None):
        return getattr(self, key, default) or default

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
