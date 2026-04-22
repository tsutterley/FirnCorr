"""
Utility functions for downloading SMB and firn model data
"""

from .fetch_mar import fetch_mar

# create fetch class to group fetching functions
fetch = type("fetch", (), {})
fetch.MAR = fetch_mar
