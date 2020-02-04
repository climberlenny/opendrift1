#!/usr/bin/env python
"""
Vietnam
==================================
"""

from datetime import datetime, timedelta
from opendrift.models.leeway import Leeway

o = Leeway(loglevel=20)  # Set loglevel to 0 for debug information

# Adding readers for global Thredds datasets:
# - Ocean forecast from global Hycom
# - Weather forecast from NOAA/NCEP
o.add_readers_from_list([
    'https://tds.hycom.org/thredds/dodsC/GLBy0.08/latest',
    'https://oos.soest.hawaii.edu/thredds/dodsC/hioos/model/atm/ncep_global/NCEP_Global_Atmospheric_Model_best.ncd'])

# Seed some particles
objType = 26  # 26 = Life-raft, no ballast
o.seed_elements(lon=107.8, lat=10.0, radius=1000, number=1000,
                objectType=objType, time=datetime.now())

# Run model
o.run(duration=timedelta(days=3),
      time_step=timedelta(hours=1),
      time_step_output=timedelta(hours=3))

# Print and plot results
print(o)
o.plot(fast=True)
o.animation(fast=True)