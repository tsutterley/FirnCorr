"""
Utility functions for downloading SMB and firn model data
"""

from .fetch_gesdisc import fetch_gesdisc
from .fetch_mar import fetch_mar

# create fetch class to group fetching functions
fetch = type("fetch", (), {})
fetch.GESDISC = fetch_gesdisc
fetch.MAR = fetch_mar
