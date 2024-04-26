# This file is part of OpenDrift.
#
# OpenDrift is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 2
#
# OpenDrift is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with OpenDrift.  If not, see <https://www.gnu.org/licenses/>.
#
# Copyright 2015, 2020, Knut-Frode Dagestad, MET Norway

import numpy as np
import math
import logging

logger = logging.getLogger(__name__)
from opendrift.models.oceandrift import OceanDrift, Lagrangian3DArray


class IcebergObj(Lagrangian3DArray):
    """Extending LagrangianArray with relevant properties for an Iceberg"""

    variables = Lagrangian3DArray.add_variables(
        [
            (
                "draft",
                {
                    "dtype": np.float32,  # Draft of Iceberg (part below sea surface) depthib
                    "units": "m",
                    "default": 90,
                },
            ),
            (
                "sail",
                {
                    "dtype": np.float32,  # Height over sea of iceberg hib
                    "units": "m",
                    "default": 10,
                },
            ),
            (
                "length",
                {
                    "dtype": np.float32,  # length of Iceberg lib
                    "units": "m",
                    "default": 100,
                },
            ),
            (
                "width",
                {
                    "dtype": np.float32,  # width of Iceberg wib
                    "units": "m",
                    "default": 30,
                },
            ),
            (
                "weight_coeff",
                {
                    "dtype": np.float32,  # weight_coeff wgt=1.0 if 100%, wgt=0.5 if 50%, wgt=0.35 if 35%, of the mass of tabular iceberg
                    "units": "1",
                    "default": 1,
                },
            ),
            (
                "water_drag_coeff",
                {"dtype": np.float32, "units": "1", "default": 0.25},  # cdo
            ),
            (
                "wind_drag_coeff",
                {"dtype": np.float32, "units": "1", "default": 0.7},  # cda
            ),
            ("iceb_x_velocity", {"dtype": np.float32, "units": "m/s", "default": 0.0}),
            ("iceb_y_velocity", {"dtype": np.float32, "units": "m/s", "default": 0.0}),
        ]
    )


class IcebergDrift(OceanDrift):

    ElementType = IcebergObj

    # Specify which environment variables (e.g. wind, waves, currents...)
    # are needed/required by the present model, to be used for updating
    # the element properties (including propagation).

    required_variables = {
        "x_sea_water_velocity": {"fallback": None, "profiles": True},
        "y_sea_water_velocity": {"fallback": None, "profiles": True},
        "sea_floor_depth_below_sea_level": {"fallback": 10_000},
        "x_wind": {"fallback": 0, "important": False},
        "y_wind": {"fallback": 0, "important": False},
        "sea_surface_wave_significant_height": {"fallback": 0},
        "sea_surface_wave_from_direction": {"fallback": 0},
        "sea_surface_wave_stokes_drift_x_velocity": {"fallback": 0},
        "sea_surface_wave_stokes_drift_y_velocity": {"fallback": 0},
        "sea_water_temperature": {"fallback": 2, "profiles": True},
        "sea_water_salinity": {"fallback": 35, "profiles": True},
        "sea_ice_area_fraction": {"fallback": 0},
        "land_binary_mask": {"fallback": None},
    }

    # Configuration
    def __init__(self, with_stokes_drift=True, wave_rad=True, *args, **kwargs):
        """3 wave models : with Stokes drift only = SD, with Radiation Force only = RF, with both = SDRF
        Apply correction when the estimated speed with the acceleration method is too big it replaces the speed by the one determinated by the no acc method
        """
        # Read Iceberg properties from external file ### will be added later

        # The constructor of parent class must always be called
        # to perform some necessary common initialisation tasks:
        super(IcebergDrift, self).__init__(*args, **kwargs)
        self.wave_rad = wave_rad
        self.with_stokes_drift = with_stokes_drift

    ########################################################################################
    def prepare_run(self):
        self.profiles_depth = self.elements_scheduled.draft.max()
        logger.info(f"Max Icebergs draft is : {self.profiles_depth}")

    def update(self):
        """Update positions and properties of particles"""

        self.advect_iceberg(self.with_stokes_drift, self.wave_rad)
        self.melt()

        # Grounding
        self.deactivate_elements(
            self.elements.draft > self.environment.sea_floor_depth_below_sea_level,
            reason="Grounded iceberg",
        )
