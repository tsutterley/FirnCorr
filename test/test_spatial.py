#!/usr/bin/env python
u"""
test_spatial.py (11/2020)
Verify spatial operations
"""
import pytest
import numpy as np
import FirnCorr.spatial

# PURPOSE: test the data type function
def test_data_type():
    # test drift type
    exp = 'drift'
    # number of data points
    npts = 30; ntime = 30
    x = np.random.rand(npts)
    y = np.random.rand(npts)
    t = np.random.rand(ntime)
    obs = FirnCorr.spatial.data_type(x,y,t)
    assert (obs == exp)
    # test grid type
    exp = 'grid'
    xgrid,ygrid = np.meshgrid(x,y)
    obs = FirnCorr.spatial.data_type(xgrid,ygrid,t)
    assert (obs == exp)
    # test grid type with spatial dimensions
    exp = 'grid'
    nx = 30; ny = 20; ntime = 10
    x = np.random.rand(nx)
    y = np.random.rand(ny)
    t = np.random.rand(ntime)
    xgrid,ygrid = np.meshgrid(x,y)
    obs = FirnCorr.spatial.data_type(xgrid,ygrid,t)
    assert (obs == exp)
    # test time series type
    exp = 'time series'
    # number of data points
    npts = 1; ntime = 1
    x = np.random.rand(npts)
    y = np.random.rand(npts)
    t = np.random.rand(ntime)
    obs = FirnCorr.spatial.data_type(x,y,t)
    assert (obs == exp)
    # test catch for unknown data type
    msg = 'Unknown data type'
    # number of data points
    npts = 30; ntime = 10
    x = np.random.rand(npts)
    y = np.random.rand(npts)
    t = np.random.rand(ntime)
    with pytest.raises(ValueError, match=msg):
        FirnCorr.spatial.data_type(x,y,t)

# PURPOSE: test ellipsoidal parameters within spatial module
def test_ellipsoid_parameters():
    assert FirnCorr.spatial._wgs84.a_axis == 6378137.0
    assert FirnCorr.spatial._wgs84.flat == (1.0/298.257223563)

# PURPOSE: test ellipsoid conversion
def test_convert_ellipsoid():
    # semimajor axis (a) and flattening (f) for TP and WGS84 ellipsoids
    atop,ftop = (6378136.3,1.0/298.257)
    awgs,fwgs = (6378137.0,1.0/298.257223563)
    # create latitude and height array in WGS84
    lat_WGS84 = 90.0 - np.arange(181,dtype=np.float64)
    elev_WGS84 = 3000.0 + np.zeros((181),dtype=np.float64)
    # convert from WGS84 to Topex/Poseidon Ellipsoids
    lat_TPX,elev_TPX = FirnCorr.spatial.convert_ellipsoid(lat_WGS84, elev_WGS84,
        awgs, fwgs, atop, ftop, eps=1e-12, itmax=10)
    # check minimum and maximum are expected from NSIDC IDL program
    # ftp://ftp.nsidc.org/DATASETS/icesat/tools/idl/ellipsoid/test_ce.pro
    minlat = np.min(lat_TPX-lat_WGS84)
    maxlat = np.max(lat_TPX-lat_WGS84)
    explat = [-1.2305653e-7,1.2305653e-7]
    minelev = 100.0*np.min(elev_TPX-elev_WGS84)
    maxelev = 100.0*np.max(elev_TPX-elev_WGS84)
    expelev = [70.000000,71.368200]
    assert np.allclose([minlat,maxlat],explat)
    assert np.allclose([minelev,maxelev],expelev)
    # convert back from Topex/Poseidon to WGS84 Ellipsoids
    phi_WGS84,h_WGS84 = FirnCorr.spatial.convert_ellipsoid(lat_TPX, elev_TPX,
        atop, ftop, awgs, fwgs, eps=1e-12, itmax=10)
    # check minimum and maximum are expected from NSIDC IDL program
    # ftp://ftp.nsidc.org/DATASETS/icesat/tools/idl/ellipsoid/test_ce.pro
    minlatdel = np.min(phi_WGS84-lat_WGS84)
    maxlatdel = np.max(phi_WGS84-lat_WGS84)
    explatdel = [-2.1316282e-14,2.1316282e-14]
    minelevdel = 100.0*np.min(h_WGS84-elev_WGS84)
    maxelevdel = 100.0*np.max(h_WGS84-elev_WGS84)
    expelevdel = [-1.3287718e-7,1.6830199e-7]
    assert np.allclose([minlatdel,maxlatdel],explatdel)
    assert np.allclose([minelevdel,maxelevdel],expelevdel,atol=1e-5)

# PURPOSE: verify cartesian to geodetic conversions
def test_convert_geodetic():
    # choose a random set of locations
    latitude = -90.0 + 180.0*np.random.rand(100)
    longitude = -180.0 + 360.0*np.random.rand(100)
    height = 2000.0*np.random.rand(100)
    # ellipsoidal parameters
    a_axis = FirnCorr.spatial._wgs84.a_axis
    flat = FirnCorr.spatial._wgs84.flat
    # convert to cartesian coordinates
    x, y, z = FirnCorr.spatial.to_cartesian(longitude, latitude, h=height,
        a_axis=a_axis, flat=flat)
    # convert back to geodetic coordinates
    ln1, lt1, h1 = FirnCorr.spatial.to_geodetic(x, y, z,
        a_axis=a_axis, flat=flat, method='moritz')
    ln2, lt2, h2 = FirnCorr.spatial.to_geodetic(x, y, z,
        a_axis=a_axis, flat=flat, method='bowring')
    ln3, lt3, h3 = FirnCorr.spatial.to_geodetic(x, y, z,
        a_axis=a_axis, flat=flat, method='zhu')
    # validate outputs for Moritz iterative method
    assert np.allclose(longitude, ln1)
    assert np.allclose(latitude, lt1)
    assert np.allclose(height, h1)
    # validate outputs for Bowring iterative method
    assert np.allclose(longitude, ln2)
    assert np.allclose(latitude, lt2)
    assert np.allclose(height, h2)
    # validate outputs for Zhu closed-form method
    assert np.allclose(longitude, ln3)
    assert np.allclose(latitude, lt3)
    assert np.allclose(height, h3)

# PURPOSE: test calculation of geocentric latitudes
def test_geocentric_latitude():
    # WGS84 ellipsoidal parameters
    a_axis = 6378137.0
    flat = (1.0/298.257223563)
    # square of the eccentricity of the ellipsoid
    ecc2 = 2.0 * flat - flat**2
    # first numerical eccentricity
    ecc1 = np.sqrt(ecc2*a_axis**2)/a_axis
    # latitude and longitude arrays for testing
    lat = 90.0 - np.arange(181, dtype=np.float64)
    lon = np.zeros((181), dtype=np.float64)
    # calculate geocentric latitudes
    test = FirnCorr.spatial.geocentric_latitude(lat, flat=flat)
    # validate against Cartesian coordinate method
    # geodetic latitude in radians
    latitude_geodetic_rad = np.radians(lat)
    # prime vertical radius of curvature
    N = a_axis/np.sqrt(1.0 - ecc1**2.*np.sin(latitude_geodetic_rad)**2.0)
    # calculate X, Y and Z from geodetic latitude and longitude
    X = N * np.cos(latitude_geodetic_rad) * np.cos(np.radians(lon))
    Y = N * np.cos(latitude_geodetic_rad) * np.sin(np.radians(lon))
    Z = (N * (1.0 - ecc1**2.0)) * np.sin(latitude_geodetic_rad)
    # calculate geocentric latitude and convert to degrees
    validation = np.degrees(np.arctan(Z / np.sqrt(X**2.0 + Y**2.0)))
    # validate outputs
    assert np.allclose(test, validation, atol=1e-5)
    # validate against Cartesian coordinate method from function
    x, y, z = FirnCorr.spatial.to_cartesian(lon, lat,
        a_axis=a_axis, flat=flat)
    # calculate geocentric latitude and convert to degrees
    validation = np.degrees(np.arctan(z / np.sqrt(x**2.0 + y**2.0)))
    # validate outputs
    assert np.allclose(test, validation, atol=1e-5)
    # compare against a different transformation
    theta = np.arctan2(
        (1.0 - ecc2) * np.sin(latitude_geodetic_rad),
        np.cos(latitude_geodetic_rad)
    )
    # validate outputs
    assert np.allclose(test, np.degrees(theta), atol=1e-5)
    assert np.allclose(np.degrees(theta), validation, atol=1e-5)

# PURPOSE: test wrap longitudes
def test_wrap_longitudes():
    # number of data points
    lon = np.arange(360)
    obs = FirnCorr.spatial.wrap_longitudes(lon)
    # expected wrapped longitudes
    exp = np.zeros((360))
    exp[:181] = np.arange(181)
    exp[181:] = np.arange(-179,0)
    assert np.allclose(obs,exp)
