=========
utilities
=========

Download and management utilities for syncing time and auxiliary files

 - Can list a directory on a ftp host
 - Can download a file from a ftp or http host
 - Can download a file from CDDIS via https when NASA Earthdata credentials are supplied
 - Checks ``MD5`` or ``sha1`` hashes between local and remote files

`Source code`__

.. __: https://github.com/tsutterley/FirnCorr/blob/main/FirnCorr/utilities.py

General Methods
===============

.. autofunction:: FirnCorr.utilities.get_data_path

.. autofunction:: FirnCorr.utilities.import_dependency

.. autofunction:: FirnCorr.utilities.get_hash

.. autofunction:: FirnCorr.utilities.url_split

.. autofunction:: FirnCorr.utilities.get_unix_time

.. autofunction:: FirnCorr.utilities.isoformat

.. autofunction:: FirnCorr.utilities.even

.. autofunction:: FirnCorr.utilities.ceil

.. autofunction:: FirnCorr.utilities.copy

.. autofunction:: FirnCorr.utilities.check_ftp_connection

.. autofunction:: FirnCorr.utilities.ftp_list

.. autofunction:: FirnCorr.utilities.from_ftp

.. autofunction:: FirnCorr.utilities.http_list

.. autofunction:: FirnCorr.utilities.from_http

.. autofunction:: FirnCorr.utilities.build_opener

.. autofunction:: FirnCorr.utilities.gesdisc_list

.. autofunction:: FirnCorr.utilities.cmr_filter_json

.. autofunction:: FirnCorr.utilities.cmr

.. autofunction:: FirnCorr.utilities.build_request
