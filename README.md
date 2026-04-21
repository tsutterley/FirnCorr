# FirnCorr

Python-based tools for correcting data for surface mass balance and firn processes

## About

<table>
  <tr>
    <td><b>Tests:</b></td>
    <td>
        <a href="https://firncorr.readthedocs.io/en/latest/?badge=latest" alt="Documentation Status"><img src="https://readthedocs.org/projects/firncorr/badge/?version=latest"></a>
        <a href="https://github.com/FirnCorr/FirnCorr/actions/workflows/python-request.yml" alt="Build"><img src="https://github.com/FirnCorr/FirnCorr/actions/workflows/python-request.yml/badge.svg"></a>
        <a href="https://github.com/FirnCorr/FirnCorr/actions/workflows/ruff-format.yml" alt="Ruff"><img src="https://github.com/FirnCorr/FirnCorr/actions/workflows/ruff-format.yml/badge.svg"></a>
    </td>
  </tr>
  <tr>
    <td><b>License:</b></td>
    <td>
        <a href="https://github.com/FirnCorr/FirnCorr/blob/main/LICENSE" alt="License"><img src="https://img.shields.io/github/license/FirnCorr/FirnCorr"></a>
    </td>
  </tr>
</table>

For more information: see the documentation at [firncorr.readthedocs.io](https://firncorr.readthedocs.io/)

## Installation

Development version from GitHub:

```bash
python3 -m pip install git+https://github.com/FirnCorr/FirnCorr.git
```

### Running with Pixi

Alternatively, you can use [Pixi](https://pixi.sh/) for a streamlined workspace environment:

1. Install Pixi following the [installation instructions](https://pixi.sh/latest/#installation)
2. Clone the project repository:

```bash
git clone https://github.com/FirnCorr/FirnCorr.git
```

3. Move into the `FirnCorr` directory

```bash
cd FirnCorr
```

4. Install dependencies and start JupyterLab:

```bash
pixi run start
```

This will automatically create the environment, install all dependencies, and launch JupyterLab in the [notebooks](./doc/source/notebooks/) directory.

## Dependencies

- [h5netcdf: Pythonic interface to netCDF4 via h5py](https://h5netcdf.org/)
- [lxml: processing XML and HTML in Python](https://pypi.python.org/pypi/lxml)
- [numpy: Scientific Computing Tools For Python](https://www.numpy.org)
- [platformdirs: Python module for determining platform-specific directories](https://pypi.org/project/platformdirs/)
- [pyproj: Python interface to PROJ library](https://pypi.org/project/pyproj/)
- [scipy: Scientific Tools for Python](https://www.scipy.org/)
- [timescale: Python tools for time and astronomical calculations](https://pypi.org/project/timescale/)
- [xarray: N-D labeled arrays and datasets in Python](https://docs.xarray.dev/en/stable/) 

## References

> B. E. Smith, B. Medley, X. Fettweis, T. Sutterley, P. Alexander, D. Porter, and M. Tedesco,
>  "Evaluating Greenland surface-mass-balance and firn-densification data using ICESat-2 altimetry",
>  *The Cryosphere*, 17(2), 789-808, (2023). [doi: 10.5194/tc-17-789-2023](https://doi.org/10.5194/tc-17-789-2023)

> T. C. Sutterley, I. Velicogna, X. Fettweis, E. Rignot, B. Noël, and M. van den Broeke,
> "Evaluation of Reconstructions of Snow/Ice Melt in Greenland by Regional Atmospheric Climate Models Using Laser Altimetry Data", *Geophysical Research Letters*, 45(16),
> 8324-8333, (2018). [doi: 10.1029/2018GL078645](https://doi.org/10.1029/2018GL078645)

## Download

The program homepage is:  
<https://github.com/tsutterley/FirnCorr>

A zip archive of the latest version is available directly at:  
<https://github.com/tsutterley/FirnCorr/archive/main.zip>

## Alternative Software

SMB correction tools built upon [`pointCollection`](https://github.com/SmithB/pointCollection):  
<https://github.com/SmithB/SMBcorr>

## Disclaimer

This package includes software developed at NASA Goddard Space Flight Center (GSFC) and the University of Washington Applied Physics Laboratory (UW-APL).
It is not sponsored or maintained by the Universities Space Research Association (USRA), AVISO or NASA.
The software is provided here for your convenience but *with no guarantees whatsoever*.

## Contributing

This project contains work and contributions from the [scientific community](./CONTRIBUTORS.md).
If you would like to contribute to the project, please have a look at the [contribution guidelines](./doc/source/getting_started/Contributing.rst), [open issues](https://github.com/FirnCorr/FirnCorr/issues) and [discussions board](https://github.com/FirnCorr/FirnCorr/discussions).

## License

The content of this project is licensed under the [Creative Commons Attribution 4.0 Attribution license](https://creativecommons.org/licenses/by/4.0/) and the source code is licensed under the [MIT license](LICENSE).
