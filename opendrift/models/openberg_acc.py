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
from opendrift.models.physics_methods import PhysicsMethods
from scipy.integrate import solve_ivp

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


########################################################################################
def water_drag(t, iceb_vel, water_vel, Ao, rho_water, water_drag_coef):
    # water velocity
    vxo, vyo = water_vel[0], water_vel[1]

    # iceberg velocity
    x_vel, y_vel = iceb_vel[0], iceb_vel[1]

    rel_water_x_vel = vxo - x_vel
    rel_water_y_vel = vyo - y_vel
    rel_water_norm = np.sqrt(rel_water_x_vel**2 + rel_water_y_vel**2)
    F_ocean_x = (
        0.5 * rho_water * water_drag_coef * Ao * rel_water_norm * rel_water_x_vel
    )
    F_ocean_y = (
        0.5 * rho_water * water_drag_coef * Ao * rel_water_norm * rel_water_y_vel
    )
    return np.array([F_ocean_x, F_ocean_y])


def wind_drag(t, iceb_vel, wind_vel, Aa, rho_air, wind_drag_coef):
    # wind velocity
    vxa, vya = wind_vel[0], wind_vel[1]

    # iceberg velocity
    x_vel, y_vel = iceb_vel[0], iceb_vel[1]

    # relative velocity
    rel_wind_x_vel = vxa - x_vel
    rel_wind_y_vel = vya - y_vel
    rel_wind_norm = np.sqrt(rel_wind_x_vel**2 + rel_wind_y_vel**2)

    F_wind_x = 0.5 * rho_air * wind_drag_coef * Aa * rel_wind_norm * rel_wind_x_vel
    F_wind_y = 0.5 * rho_air * wind_drag_coef * Aa * rel_wind_norm * rel_wind_y_vel
    return np.array([F_wind_x, F_wind_y])


def wave_radiation_force(
    t,
    iceb_vel,
    rho_water,
    wave_drag_coef,
    g,
    wave_height,
    wave_direction,
    iceb_length,
):

    F_wave_x = (
        0.5
        * rho_water
        * wave_drag_coef
        * g
        * iceb_length
        * (wave_height / 2) ** 2
        * np.sin(
            np.deg2rad((wave_direction + 180) % 360)
        )  # (angle+180)%360 because the angle indicates where does the wave come from
    )
    F_wave_y = (
        0.5
        * rho_water
        * wave_drag_coef
        * g
        * iceb_length
        * (wave_height / 2) ** 2
        * np.cos(np.deg2rad((wave_direction + 180) % 360))
    )
    return np.array([F_wave_x, F_wave_y])


def advect_iceberg_no_acc(f, water_vel, wind_vel):

    vxo, vyo = water_vel[0], water_vel[1]
    vxa, vya = wind_vel[0], wind_vel[1]

    no_acc_model_x = (1 - f) * vxo + f * vxa
    no_acc_model_y = (1 - f) * vyo + f * vya
    return np.array([no_acc_model_x, no_acc_model_y])


# MELTING ###################################################################
def melwav(lib, wib, uaib, vaib, sst, conc, dt):
    """update length and width value according to wave melting

    Args:
        lib (_type_): iceberg length
        wib (_type_): iceberg width
        uaib (_type_): wind speed u component
        vaib (_type_): wind speed v component
        sst (_type_): Sea surface temperature
        conc (_type_): sea ice concentration
    """
    Ss = -5 + np.sqrt(32 + 2 * np.sqrt(uaib**2 + vaib**2))
    Vsst = (1 / 6.0) * (sst + 2) * Ss
    Vwe = Vsst * 0.5 * (1 + np.cos(np.pi * conc**3))  # melting in m/day ?
    Vwe /= 86400  # melting in m/s ?

    # length lost only on one side
    new_lib = np.zeros_like(lib)
    new_wib = np.zeros_like(wib)
    new_lib[lib != 0] = lib[lib != 0] - Vwe[lib != 0] * dt
    new_wib[lib != 0] = wib[lib != 0] / lib[lib != 0] * new_lib[lib != 0]

    return new_lib, new_wib


def mellat(lib, wib, tempib, salnib, dt):
    # Lateral melting parameterization taken from Kubat et al. 2007
    # An operational iceberg deterioration model
    """_summary_

    Args:
        lib (_type_): iceberg length
        wib (_type_): icebrg width
        tempib (_type_): far field water temperature vector size nz*N_elements
        salnib (_type_): water salinity vector size nz*N_elements
        dt : timestep in second
    """
    TfS = -0.036 - 0.0499 * salnib - 0.000112 * salnib**2
    Tfp = TfS * np.exp(-0.19 * (tempib - TfS))
    deltaT = tempib - Tfp
    deltaT = np.concatenate([deltaT, deltaT**2], axis=0)
    coefs = np.concatenate(
        [np.ones_like(tempib) * 2.78, np.ones_like(tempib) * 0.47], axis=0
    )
    sumVb = np.diag(np.dot(deltaT.T, coefs))

    # Unit of sumVb [meter/year]
    # Convert to meter per second
    dx = sumVb / 365 / 86400 * dt

    # Change of iceberg length (on both sides?? -> 2.*)

    new_lib = np.zeros_like(lib)
    new_wib = np.zeros_like(wib)
    new_lib[lib != 0] = lib[lib != 0] - 2 * dx[lib != 0]
    new_wib[lib != 0] = (
        wib[lib != 0] / lib[lib != 0] * new_lib[lib != 0]
    )  # keep the same width/length ratio
    return new_lib, new_wib


def melbas(depthib, sailib, lib, salnib, tempib, uoib, voib, uib, vib, dt):
    """
    Calculate the surface melt due to forced convection following Kubat et al.
    (An operational iceberg deterioration Model 2007).

    :param depthib: Iceberg depth (positive value)
    :param lib: Iceberg length scale
    :param salnib: Salinity profile in the ocean (array of length nz)
    :param tempib: Temperature profile in the ocean (array of length nz)
    :param uoib: Ocean velocity components in the x-direction (array of length nz)
    :param voib: Ocean velocity components in the y-direction (array of length nz)
    :param uib: Iceberg drift velocity in the x-direction
    :param vib: Iceberg drift velocity in the y-direction

    The function updates the depthib parameter.
    """

    # Temperature at the base of the iceberg
    absv = np.sqrt(((uoib[-1] - uib) ** 2 + (voib[-1] - vib) ** 2))
    TfS = -0.036 - 0.0499 * salnib[-1] - 0.000112 * salnib[-1] ** 2
    Tfp = TfS * 2.71828 ** (-0.19 * (tempib[-1] - TfS))
    deltat = tempib[-1] - Tfp

    Vf = 0.58 * absv**0.8 * deltat / (lib**0.2)
    Vf = Vf / 86400  # convert in m/s

    # Archimedean buoyant force
    T_mean = tempib.mean(axis=0)
    S_mean = salnib.mean(axis=0)
    rho_water = PhysicsMethods.sea_water_density(T_mean, S_mean)
    rho_iceb = 900
    factor = rho_iceb / rho_water
    H = depthib + sailib
    depthib = H * factor
    sailib = H - depthib

    # Update the depth
    new_depthib = np.zeros_like(depthib)
    new_sailib = np.zeros_like(sailib)
    new_depthib[depthib != 0] = (
        abs(depthib[depthib != 0]) - Vf[depthib != 0] * dt
    )  # melt the iceberg base

    # finding the new force balance according to Archimedean buoyant force
    H = new_depthib + sailib
    new_depthib = H * factor
    new_sailib = H - new_depthib

    return new_depthib, new_sailib


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

    def advect_iceberg(self, stokes_drift=True, wave_rad=True):

        # Constants
        rho_water = 1027
        rho_air = 1.293
        rho_iceb = 900
        wave_drag_coef = 0.3
        g = 9.81
        Ao = abs(self.elements.draft) * self.elements.length  ### Area_wet
        Aa = self.elements.sail * self.elements.length  ### Area_dry
        mass = self.elements.width * (Aa + Ao) * rho_iceb  # volume * rho,  [kg]
        k = (
            rho_air
            * self.elements.wind_drag_coeff
            * Aa
            / (rho_water * self.elements.water_drag_coeff * Ao)
        )
        f = np.sqrt(k) / (1 + np.sqrt(k))

        # environment
        water_vel = np.array(
            [
                self.environment.x_sea_water_velocity
                + int(stokes_drift)
                * self.environment.sea_surface_wave_stokes_drift_x_velocity,
                self.environment.y_sea_water_velocity
                + int(stokes_drift)
                * self.environment.sea_surface_wave_stokes_drift_y_velocity,
            ]
        )
        wind_vel = np.array([self.environment.x_wind, self.environment.y_wind])
        wave_height = self.environment.sea_surface_wave_significant_height
        wave_direction = self.environment.sea_surface_wave_from_direction

        def dynamic(
            t,
            V,
            water_vel,
            wind_vel,
            wave_height,
            wave_direction,
            Ao,
            Aa,
            rho_water,
            rho_air,
            water_drag_coef,
            wind_drag_coef,
            wave_drag_coef,
            g,
            L,
            mass,
        ):
            sum_force = (
                water_drag(t, V, water_vel, Ao, rho_water, water_drag_coef)
                + wind_drag(t, V, wind_vel, Aa, rho_air, wind_drag_coef)
                + int(wave_rad)
                * wave_radiation_force(
                    t,
                    V,
                    rho_water,
                    wave_drag_coef,
                    g,
                    wave_height,
                    wave_direction,
                    L,
                )
            )
            return 1 / mass * sum_force

        V0 = advect_iceberg_no_acc(f, water_vel, wind_vel).flatten()
        sol = solve_ivp(
            dynamic,
            [0, self.time_step.total_seconds()],
            V0,
            args=(
                water_vel,
                wind_vel,
                wave_height,
                wave_direction,
                Ao,
                Aa,
                rho_water,
                rho_air,
                self.elements.water_drag_coeff,
                self.elements.wind_drag_coeff,
                wave_drag_coef,
                g,
                self.elements.length,
                mass,
            ),
            vectorized=True,
            t_eval=np.array([self.time_step.total_seconds()]),
        )
        V = sol.y.reshape((2, -1))
        Vx, Vy = V[0], V[1]
        self.update_positions(Vx, Vy)
        self.elements.iceb_x_velocity, self.elements.iceb_y_velocity = Vx, Vy

    def melt(self):
        # loading the required variables :
        x_wind = self.environment.x_wind
        y_wind = self.environment.y_wind
        uoib = self.environment_profiles["x_sea_water_velocity"]
        voib = self.environment_profiles["y_sea_water_velocity"]
        T_profile = self.environment_profiles["sea_water_temperature"]
        S_profile = self.environment_profiles["sea_water_salinity"]
        ice_conc = self.environment.sea_ice_area_fraction

        # wave melting
        self.elements.length, self.elements.width = melwav(
            self.elements.length,
            self.elements.width,
            x_wind,
            y_wind,
            T_profile[0],
            ice_conc,
            self.time_step.total_seconds(),
        )
        # lateral melting
        self.elements.length, self.elements.width = mellat(
            self.elements.length,
            self.elements.width,
            T_profile,
            S_profile,
            self.time_step.total_seconds(),
        )
        # basal melting
        self.elements.draft, self.elements.sail = melbas(
            self.elements.draft,
            self.elements.sail,
            self.elements.length,
            S_profile,
            T_profile,
            uoib,
            voib,
            self.elements.iceb_x_velocity,
            self.elements.iceb_y_velocity,
            self.time_step.total_seconds(),
        )

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
