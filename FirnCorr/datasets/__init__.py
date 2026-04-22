"""
Utility functions for downloading SMB and firn model data
"""

from .fetch_mar_ftp import fetch_mar_ftp

# create fetch class to group fetching functions
fetch = type("fetch", (), {})
fetch.MAR = fetch_mar_ftp
