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
            ("x_vel", {"dtype": np.float32, "units": "m/s", "default": 0}),
            ("y_vel", {"dtype": np.float32, "units": "m/s", "default": 0}),
            (
                "derivation_timestep",
                {"dtype": np.float32, "units": "s", "default": 900},
            ),
        ]
    )


class IcebergDrift(OceanDrift):

    ElementType = IcebergObj

    # Specify which environment variables (e.g. wind, waves, currents...)
    # are needed/required by the present model, to be used for updating
    # the element properties (including propagation).

    required_variables = {
        "x_sea_water_velocity": {"fallback": None},
        "y_sea_water_velocity": {"fallback": None},
        "sea_floor_depth_below_sea_level": {"fallback": 10_000},
        "x_wind": {"fallback": 0, "important": False},
        "y_wind": {"fallback": 0, "important": False},
        "sea_surface_wave_significant_height": {"fallback": 0},
        "sea_surface_wave_from_direction": {"fallback": 0},
        "sea_surface_wave_stokes_drift_x_velocity": {"fallback": 0},
        "sea_surface_wave_stokes_drift_y_velocity": {"fallback": 0},
        "land_binary_mask": {"fallback": None},
    }

    # Configuration
    def __init__(self, wave_model, *args, **kwargs):
        """3 wave models : with Stokes drift only = SD, with Radiation Force only = RF, with both = SDRF"""
        # Read Iceberg properties from external file ### will be added later

        # The constructor of parent class must always be called
        # to perform some necessary common initialisation tasks:
        super(IcebergDrift, self).__init__(*args, **kwargs)
        self.wave_model = wave_model

    ########################################################################################

    def update(self):
        """Update positions and properties of particles"""
        # Constants
        rho_water = 1027
        rho_air = 1.293
        rho_ice = 917
        rho_iceb = 900
        wave_drag_coef = 0.3
        g = 9.81

        # Areas exposed
        Ao = abs(self.elements.draft) * self.elements.length  ### Area_wet
        Aa = self.elements.sail * self.elements.length  ### Area_dry

        mass = self.elements.width * (Aa + Ao) * rho_iceb  # volume * rho,  [kg]
        dt = self.elements.derivation_timestep
        k = (
            rho_air
            * self.elements.wind_drag_coeff
            * Aa
            / (rho_water * self.elements.water_drag_coeff * Ao)
        )
        f = np.sqrt(k) / (1 + np.sqrt(k))

        vxo = self.environment.x_sea_water_velocity
        vyo = self.environment.y_sea_water_velocity
        x_stokes_drift = self.environment.sea_surface_wave_stokes_drift_x_velocity
        y_stokes_drift = self.environment.sea_surface_wave_stokes_drift_y_velocity
        if self.wave_model in ["SD", "SDRF"]:
            vxo = vxo + x_stokes_drift
            vyo = vyo + y_stokes_drift

        vxa = self.environment.x_wind
        vya = self.environment.y_wind

        x_vel = self.elements.x_vel
        y_vel = self.elements.y_vel

        rel_water_x_vel = vxo - x_vel
        rel_water_y_vel = vyo - y_vel
        rel_water_norm = np.sqrt(rel_water_x_vel**2 + rel_water_y_vel**2)

        rel_wind_x_vel = vxa - x_vel
        rel_wind_y_vel = vya - y_vel
        rel_wind_norm = np.sqrt(rel_wind_x_vel**2 + rel_wind_y_vel**2)

        # Ocean
        F_ocean_x = (
            0.5
            * rho_water
            * self.elements.water_drag_coeff
            * Ao
            * rel_water_norm
            * rel_water_x_vel
        )
        F_ocean_y = (
            0.5
            * rho_water
            * self.elements.water_drag_coeff
            * Ao
            * rel_water_norm
            * rel_water_y_vel
        )

        # Wind
        F_wind_x = (
            0.5
            * rho_air
            * self.elements.wind_drag_coeff
            * Aa
            * rel_wind_norm
            * rel_wind_x_vel
        )
        F_wind_y = (
            0.5
            * rho_air
            * self.elements.wind_drag_coeff
            * Aa
            * rel_wind_norm
            * rel_wind_y_vel
        )

        # Waves
        with_waves = self.wave_model in ["RF", "SDRF"]
        F_wave_x = int(with_waves) * (
            0.5
            * rho_water
            * wave_drag_coef
            * g
            * self.elements.length
            * (self.environment.sea_surface_wave_significant_height / 2) ** 2
            * np.sin(np.deg2rad(self.environment.sea_surface_wave_from_direction))
        )
        F_wave_y = int(with_waves) * (
            0.5
            * rho_water
            * wave_drag_coef
            * g
            * self.elements.length
            * (self.environment.sea_surface_wave_significant_height / 2) ** 2
            * np.cos(np.deg2rad(self.environment.sea_surface_wave_from_direction))
        )

        # Update velocities with Eulerian sheme with the Total Forces
        x_vel_tot = x_vel + dt / mass * (F_ocean_x + F_wind_x)
        no_acc_model = (1 - f) * vxo + f * vxa
        try:
            x_vel_tot[np.where(x_vel_tot >= 0)] = 0.5 * (
                x_vel_tot + no_acc_model - np.abs(x_vel_tot - no_acc_model)
            )  # min stability
        except ValueError:
            pass
        try:
            x_vel_tot[np.where(x_vel_tot < 0)] = 0.5 * (
                x_vel_tot + no_acc_model + np.abs(x_vel_tot - no_acc_model)
            )  # max
        except ValueError:
            pass
        x_vel_tot = x_vel_tot + dt / mass * (F_wave_x)

        y_vel_tot = y_vel + dt / mass * (F_ocean_y + F_wind_y + F_wave_y)
        no_acc_model = (1 - f) * vyo + f * vya
        try:
            y_vel_tot[np.where(y_vel_tot >= 0)] = 0.5 * (
                y_vel_tot + no_acc_model - np.abs(y_vel_tot - no_acc_model)
            )  # min stability
        except ValueError:
            pass
        try:
            y_vel_tot[np.where(y_vel_tot < 0)] = 0.5 * (
                y_vel_tot + no_acc_model + np.abs(y_vel_tot - no_acc_model)
            )  # max
        except ValueError:
            pass
        y_vel_tot = y_vel_tot + dt / mass * (F_wave_y)

        tot_vel = np.sqrt(x_vel_tot**2 + y_vel_tot**2)
        if np.any(tot_vel > 2):
            logger.warning(f"iceberg speed too important : {np.max(tot_vel)}")

        self.elements.x_vel = x_vel_tot
        self.elements.y_vel = y_vel_tot
        self.update_positions(x_vel_tot, y_vel_tot)

        # Grounding
        self.deactivate_elements(
            self.elements.draft > self.environment.sea_floor_depth_below_sea_level,
            reason="Grounded iceberg",
        )
