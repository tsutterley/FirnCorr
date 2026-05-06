"""
Utility functions for downloading SMB and firn model data
"""

from .fetch_gemb import fetch_gemb
from .fetch_gesdisc import fetch_gesdisc
from .fetch_gsfcfdm import fetch_gsfcfdm
from .fetch_mar import fetch_mar

# create fetch class to group fetching functions
fetch = type("fetch", (), {})
fetch.GEMB = fetch_gemb
fetch.GESDISC = fetch_gesdisc
fetch.GSFCfdm = fetch_gsfcfdm
fetch.MAR = fetch_mar
