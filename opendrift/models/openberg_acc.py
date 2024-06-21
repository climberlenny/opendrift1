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
# Copyright 2024, Lenny Hucher, NERSC

import numpy as np
import logging
from opendrift.models.physics_methods import PhysicsMethods
from scipy.integrate import solve_ivp
import itertools

logger = logging.getLogger(__name__)
from opendrift.models.oceandrift import OceanDrift, Lagrangian3DArray


# GLOBAL CONSTANTS
rho_air = 1.293
rho_iceb = 900
rhosi = 917
wave_drag_coef = 0.3
g = 9.81
csi = 1  # sea ice coefficient of resistance


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


# FORCING #######################################################################################


def water_drag(t, iceb_vel, water_vel, Ao, rho_water, water_drag_coef):
    """taken from Keghouche et al. 2009

    Args:
        t : dummy parameter corresponding to the current time
        iceb_vel : Iceberg velocity at time t
        water_vel : current velocity [m/s]
        Ao : iceberg area exposed to the current = length x draft
        rho_water : water volumic mass [kg/m3]
        water_drag_coef : Co = drag coefficient applied on the draft of the iceberg

    Returns:
        water drag force
    """
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
    """taken from Keghouche et al. 2009
    Args:
        t : dummy parameter corresponding to the current time
        iceb_vel : Iceberg velocity at time t
        wind_vel : wind velocity [m/s]
        Aa : iceberg area exposed to the wind = length x sail
        rho_air : air volumic mass [kg/m3]
        wind_drag_coef : Ca = drag coefficient applied on the sail of the iceberg

    Returns:
        wind drag force
    """
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
    """
    Args:
        t : dummy parameter corresponding to the current time
        iceb_vel : Iceberg velocity at time t
        rho_water : water volumic mass [kg/m3]
        wave_drag_coef : constant
        g : constant
        wave_height : wave significant height
        wave_direction : wave direction
        iceb_length : iceberg length

    Returns:
        wave radiation force
    """
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
    """advect the iceberg as a drifter by approximating its acceleration to be null

    Args:
        f : wind drift factor
        water_vel : current velocity
        wind_vel : wind velocity

    Returns:
        iceberg velocity approximated
    """
    vxo, vyo = water_vel[0], water_vel[1]
    vxa, vya = wind_vel[0], wind_vel[1]

    no_acc_model_x = (1 - f) * vxo + f * vxa
    no_acc_model_y = (1 - f) * vyo + f * vya
    V = np.array([no_acc_model_x, no_acc_model_y])
    if not np.isfinite(V).all():
        logger.error(
            "infinite value in approximated iceberg velocity : check the wind drift factor f values"
        )
    return V


def sea_ice_force(
    t,
    iceb_vel,
    sea_ice_conc,
    sea_ice_thickness,
    sea_ice_vel,
    rhosi,
    iceb_width,
    sum_force,
):
    """taken from Keghouche et al. 2009
    Args:
        t : dummy parameter corresponding to the current time
        iceb_vel : Iceberg velocity at time t
        sea_ice_conc : sea ice concentration [%]
        sea_ice_thickness : sea ice thickness [m]
        sea_ice_vel : sea ice velocity [m/s]
        rhosi : sea ice volumic mass [kg/m3]
        iceb_width : iceberg width
        sum_force : sum of the other force affecting the iceberg excluding the sea ice force

    Returns:
        sea ice force
    """

    ice_x, ice_y = sea_ice_vel
    x_vel, y_vel = iceb_vel[0], iceb_vel[1]
    diff_vel = np.sqrt((ice_x - x_vel) ** 2 + (ice_y - y_vel) ** 2)
    force_x, force_y = sum_force
    F_ice_x = np.zeros_like(x_vel)
    F_ice_y = np.zeros_like(y_vel)
    F_ice_x = (
        0.5
        * (rhosi * csi * sea_ice_thickness * iceb_width)
        * diff_vel
        * (ice_x - x_vel)
    )
    F_ice_y = (
        0.5
        * (rhosi * csi * sea_ice_thickness * iceb_width)
        * diff_vel
        * (ice_y - y_vel)
    )
    F_ice_x[sea_ice_conc <= 0.15] = 0
    F_ice_y[sea_ice_conc <= 0.15] = 0
    F_ice_x[sea_ice_conc >= 0.9] = -force_x[sea_ice_conc >= 0.9]
    F_ice_y[sea_ice_conc >= 0.9] = -force_y[sea_ice_conc >= 0.9]
    return np.array([F_ice_x, F_ice_y])


# MELTING ###################################################################
def melwav(iceb_length, iceb_width, x_wind, y_wind, sst, conc, dt):
    """update length and width value according to wave melting. taken from Keghouche et al. 2009

    Args:
        iceb_length : iceberg length
        iceb_width : iceberg width
        x_wind : wind speed x component
        y_wind : wind speed y component
        sst : Sea surface temperature
        conc : sea ice concentration
        dt : timestep of the simulation [s]
    """
    Ss = -5 + np.sqrt(32 + 2 * np.sqrt(x_wind**2 + y_wind**2))
    Vsst = (1 / 6.0) * (sst + 2) * Ss
    Vwe = Vsst * 0.5 * (1 + np.cos(np.pi * conc**3)) / 86400  # melting in m/s to Check

    # length lost only on one side
    new_iceb_length = np.zeros_like(iceb_length)
    new_iceb_width = np.zeros_like(iceb_width)
    new_iceb_length[iceb_length != 0] = (
        iceb_length[iceb_length != 0] - Vwe[iceb_length != 0] * dt
    )
    new_iceb_width[iceb_length != 0] = (
        iceb_width[iceb_length != 0]
        / iceb_length[iceb_length != 0]
        * new_iceb_length[iceb_length != 0]
    )
    new_iceb_length[new_iceb_length < 0] = 0
    new_iceb_width[new_iceb_width < 0] = 0
    return new_iceb_length, new_iceb_width


def mellat(iceb_length, iceb_width, tempib, salnib, dt):
    # Lateral melting parameterization taken from Kubat et al. 2007
    # An operational iceberg deterioration model
    """update length and width value according to wave melting. taken from Keghouche et al. 2009

    Args:
        iceb_length : iceberg length
        iceb_width : iceberg width
        tempib : far field water temperature vector size nz*N_elements
        salnib : water salinity vector size nz*N_elements
        dt : timestep of the simulation [s]
    """
    TfS = -0.036 - 0.0499 * salnib - 0.000112 * salnib**2
    Tfp = TfS * np.exp(-0.19 * (tempib - TfS))
    deltaT = tempib - Tfp
    deltaT = np.concatenate([2.78 * deltaT, 0.47 * deltaT**2], axis=0)
    sumVb = np.nansum(deltaT, axis=0)

    # Unit of sumVb [meter/year]
    # Convert to meter per second
    dx = sumVb / 365 / 86400 * dt

    # Change of iceberg length (on both sides?? -> 2.*)

    new_iceb_length = np.zeros_like(iceb_length)
    new_iceb_width = np.zeros_like(iceb_width)
    new_iceb_length[iceb_length != 0] = (
        iceb_length[iceb_length != 0] - 2 * dx[iceb_length != 0]
    )
    new_iceb_width[iceb_length != 0] = (
        iceb_width[iceb_length != 0]
        / iceb_length[iceb_length != 0]
        * new_iceb_length[iceb_length != 0]
    )
    # keep the same width/length ratio
    new_iceb_length[new_iceb_length < 0] = 0
    new_iceb_width[new_iceb_width < 0] = 0
    return new_iceb_length, new_iceb_width


def melbas(
    iceb_draft,
    iceb_sail,
    iceb_length,
    salnib,
    tempib,
    x_water_vel,
    y_water_vel,
    x_iceb_vel,
    y_iceb_vel,
    dt,
):
    """Calculate the surface melt due to forced convection following Kubat et al.
    (An operational iceberg deterioration Model 2007). The function updates the iceb_draft parameter.

    Args:
        iceb_draft (_type_): iceberg draft
        iceb_sail (_type_): iceberg sail
        iceb_length (_type_): iceberg length
        salnib (_type_): sea water salinity at the basal layer
        tempib (_type_): sea water temperature at the basal layer
        x_water_vel (_type_): x sea water velocity at the basal layer
        y_water_vel (_type_): y sea water velocity at the basal layer
        x_iceb_vel (_type_): x iceberg velocity
        y_iceb_vel (_type_): y iceberg velocity
        dt (_type_): timestep of the simulation [s]

    """

    # Temperature at the basal layer of the iceberg
    absv = np.sqrt(((x_water_vel - x_iceb_vel) ** 2 + (y_water_vel - y_iceb_vel) ** 2))
    TfS = -0.036 - 0.0499 * salnib - 0.000112 * salnib**2
    Tfp = TfS * 2.71828 ** (-0.19 * (tempib - TfS))
    deltat = tempib - Tfp

    Vf = 0.58 * absv**0.8 * deltat / (iceb_length**0.2)
    Vf = Vf / 86400  # convert in m/s

    # Update the depth
    new_iceb_draft = np.zeros_like(iceb_draft)
    new_iceb_draft[iceb_draft != 0] = (
        abs(iceb_draft[iceb_draft != 0]) - Vf[iceb_draft != 0] * dt
    )
    # melt the iceberg base
    new_iceb_draft[iceb_draft < 0] = 0

    return new_iceb_draft, iceb_sail


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
        "sea_ice_thickness": {"fallback": 0},
        "sea_ice_x_velocity": {"fallback": 0, "important": False},
        "sea_ice_y_velocity": {"fallback": 0, "important": False},
        "land_binary_mask": {"fallback": None},
    }

    def get_profile_masked(self, variable):
        """mask profile to keep the data only for the iceberg draft

        Args:
            variable : the variable you need the profile to be masked

        Returns:
            The masked profile of the variable for each iceberg of the simulation
        """
        draft = self.elements.draft
        profile = self.environment_profiles[variable]
        z = self.environment_profiles["z"]
        mask = draft[:, np.newaxis] < -z
        mask = mask.T
        mask[np.argmax(mask, axis=0), np.arange(mask.shape[1])] = False
        return np.ma.masked_array(profile, mask, fill_value=np.nan)

    def get_basal_env(self, variable):
        """get the basal layer of the variable for each iceberg

        Args:
            variable (_type_): the variable you need the profile to be masked

        """
        profile = self.get_profile_masked(variable)
        last = np.argmin(np.logical_not(profile.mask), axis=0) - 1
        return profile[last, np.arange(profile.shape[1])]

    # TODO Check seed ensemble
    def seed_ensemble(
        self,
        Ca: tuple,
        Co: tuple,
        drag_coeff_distribution="normal",
        **kwargs,
    ):
        """_summary_

        Args:
            Ca (tuple): (min,max,mean,std)
            Co (tuple): (min,max,mean,std)
            numbers (tuple, optional): number of distinct member per param. Defaults to (10,10,10,10,10).

        Returns:
            _type_: _description_
        """
        # TODO clean non random part
        if len(Ca) == 2:
            Ca_min, Ca_max = 0.5, 2.5
            logger.debug(
                f"Default min and max value for wind drag coeff are {Ca_min} and {Ca_max}"
            )
            Ca_mean, Ca_std = Ca
        elif len(Ca) == 4:
            Ca_min, Ca_max, Ca_mean, Ca_std = Ca
        else:
            logger.warning(f"Ca wrong size. Please give (min,max,mean,std).")
        if len(Co) == 2:
            Co_min, Co_max = 0.5, 2.5
            logger.debug(
                f"Default min and max value for wind drag coeff are {Co_min} and {Co_max}"
            )
            Co_mean, Co_std = Co
        elif len(Co) == 4:
            Co_min, Co_max, Co_mean, Co_std = Co
        else:
            logger.warning(f"Co wrong size. Please give (min,max,mean,std).")
        if Ca_min > Ca_mean:
            logger.warning(
                f"Ca mean value too small compare to min : mean={Ca_mean} , min={Ca_min}"
            )
        if Co_min > Co_mean:
            logger.error(
                f"Co mean value too small compare to min : mean={Co_mean} , min={Co_min}"
            )

        if drag_coeff_distribution == "normal":
            Ens_Ca = np.abs(np.random.normal(Ca_mean, Ca_std, 10_000))
            Ens_Co = np.abs(np.random.normal(Co_mean, Co_std, 10_000))
        elif drag_coeff_distribution == "uniform":
            Ens_Ca = np.abs(
                np.random.uniform(Ca_mean - Ca_std, Ca_mean + Ca_std, 10_000)
            )
            Ens_Co = np.abs(
                np.random.uniform(Co_mean - Co_std, Co_mean + Co_std, 10_000)
            )
        if drag_coeff_distribution == "normal":
            # filter Ca and Co to have a normal distribution between min an max values
            filtered_Ca = Ens_Ca[(Ens_Ca >= Ca_min) & (Ens_Ca <= Ca_max)]
            while len(filtered_Ca) < 10_000:
                extra_values = np.random.normal(Ca_mean, Ca_std, 10_000)
                filtered_Ca = np.concatenate(
                    (
                        filtered_Ca,
                        extra_values[
                            (extra_values >= Ca_min) & (extra_values <= Ca_max)
                        ],
                    )
                )
            filtered_Ca = filtered_Ca[:10_000]

            filtered_Co = Ens_Co[(Ens_Co >= Co_min) & (Ens_Co <= Co_max)]
            while len(filtered_Co) < 10_000:
                extra_values = np.random.normal(Co_mean, Co_std, 10_000)
                filtered_Co = np.concatenate(
                    (
                        filtered_Co,
                        extra_values[
                            (extra_values >= Co_min) & (extra_values <= Co_max)
                        ],
                    )
                )
            filtered_Co = filtered_Co[:10_000]
            Ens_Ca, Ens_Co = filtered_Ca, filtered_Co
        combined_array = np.vstack((Ens_Ca, Ens_Co))
        combined_array = combined_array.T  # shape(10_000,2)
        np.random.shuffle(combined_array)
        if "number" in kwargs:
            number = kwargs["number"]
        else:
            number = 100
        sampled_array = combined_array[:number]
        logger.info("seeding ensemble ...")
        ca = sampled_array[:, 0]
        co = sampled_array[:, 1]
        self.seed_elements(
            **kwargs,
            water_drag_coeff=co,
            wind_drag_coeff=ca,
        )
        return 0

    def seed_ensemble2(
        self,
        width: tuple,
        height: tuple,
        Ca: dict = {"min": 0.1, "max": 1.5},
        ratio: tuple = (2.25, 5, 10),
        drag_coeff_distribution="beta",
        **kwargs,
    ):

        w_mean, w_std = width
        h_mean, h_std = height

        if drag_coeff_distribution == "beta":
            a, b, c = ratio
            ratio = np.random.beta(a, b, 10000) * c
            Ca_min, Ca_max = Ca
            Ca = np.random.uniform(0.1, 1.5, 10000)
            Co = Ca / ratio
        else:
            raise ValueError()

        Ens_w = np.abs(np.random.normal(w_mean, w_std, 10_000))
        Ens_h = np.abs(np.random.normal(h_mean, h_std, 10_000))
        if drag_coeff_distribution == "normal":
            Ens_Ca = np.abs(np.random.normal(Ca_mean, Ca_std, 10_000))
            Ens_Co = np.abs(np.random.normal(Co_mean, Co_std, 10_000))
        elif drag_coeff_distribution == "uniform":
            Ens_Ca = np.abs(
                np.random.uniform(Ca_mean - Ca_std, Ca_mean + Ca_std, 10_000)
            )
            Ens_Co = np.abs(
                np.random.uniform(Co_mean - Co_std, Co_mean + Co_std, 10_000)
            )

        if drag_coeff_distribution == "normal":
            # filter Ca and Co to have a normal distribution between min an max values
            filtered_Ca = Ens_Ca[(Ens_Ca >= Ca_min) & (Ens_Ca <= Ca_max)]
            while len(filtered_Ca) < 10_000:
                extra_values = np.random.normal(Ca_mean, Ca_std, 10_000)
                filtered_Ca = np.concatenate(
                    (
                        filtered_Ca,
                        extra_values[
                            (extra_values >= Ca_min) & (extra_values <= Ca_max)
                        ],
                    )
                )
            filtered_Ca = filtered_Ca[:10_000]

            filtered_Co = Ens_Co[(Ens_Co >= Co_min) & (Ens_Co <= Co_max)]
            while len(filtered_Co) < 10_000:
                extra_values = np.random.normal(Co_mean, Co_std, 10_000)
                filtered_Co = np.concatenate(
                    (
                        filtered_Co,
                        extra_values[
                            (extra_values >= Co_min) & (extra_values <= Co_max)
                        ],
                    )
                )
            filtered_Co = filtered_Co[:10_000]
            Ens_Ca, Ens_Co = filtered_Ca, filtered_Co

        rho_iceb = 900
        rho_water = 1_000
        alpha = rho_iceb / rho_water
        crit = np.sqrt(6 * alpha * (1 - alpha))

        combined_array = np.vstack((Ens_w, Ens_h, Ens_Ca, Ens_Co))
        combined_array = combined_array.T  # shape(10_000,4)
        np.random.shuffle(combined_array)
        if "number" in kwargs:
            number = kwargs["number"]
        else:
            number = np.prod(numbers)
        sampled_array = combined_array[:number]
        logger.info("seeding ensemble ...")
        w = sampled_array[:, 0]
        h = sampled_array[:, 1]
        ca = sampled_array[:, 2]
        co = sampled_array[:, 3]
        l = w
        draft = h * alpha
        sail = h - draft
        self.seed_elements(
            **kwargs,
            width=w,
            length=l,
            draft=draft,
            sail=sail,
            water_drag_coeff=co,
            wind_drag_coeff=ca,
        )
        return 0

    # Configuration
    def __init__(
        self,
        add_stokes_drift: bool = True,
        wave_rad: bool = True,
        grounding: bool = False,
        vertical_profile: bool = False,
        melting: bool = False,
        choose_melting: dict[bool] = {"wave": True, "lateral": True, "basal": True},
        *args,
        **kwargs,
    ):

        # Read Iceberg properties from external file ### will be added later

        # The constructor of parent class must always be called
        # to perform some necessary common initialisation tasks:
        super(IcebergDrift, self).__init__(*args, **kwargs)
        self.wave_rad = wave_rad
        self.add_stokes_drift = add_stokes_drift
        self.grounding = grounding
        self.vertical_profile = vertical_profile
        self.melting = (
            melting  # boolean parameter to decide wether or not the iceberg melt
        )
        self.choose_melting = (
            choose_melting  # boolean dictionnary to decide how the iceberg melt
        )

    def advect_iceberg(
        self,
        rho_water=1027,
        stokes_drift=True,
        wave_rad=True,
        grounding=False,
        vertical_profile=False,
    ):
        """Main function to advect the iceberg according to the different forcings

        Args:
            rho_water (int, optional): sea water volumic mass [kg/m3]. Defaults to 1027.
            stokes_drift (bool, optional): boolean to decide wether or not we add the stokes drift to the current velocity. Defaults to True.
            wave_rad (bool, optional): boolean to decide wether or not we add the wave radiation forcing. Defaults to True.
            grounding (bool, optional): boolean to decide wether or not we take into account of the grounding of the iceberg. Defaults to False.
            vertical_profile (bool, optional): boolean to decide wether or not we integrate the current velocity over the iceberg draft. Defaults to False.
        """
        draft = self.elements.draft
        length = self.elements.length
        Ao = abs(draft) * length  ### Area_wet
        Aa = self.elements.sail * length  ### Area_dry
        mass = self.elements.width * (Aa + Ao) * rho_iceb  # volume * rho,  [kg]
        k = (
            rho_air
            * self.elements.wind_drag_coeff
            * Aa
            / (rho_water * self.elements.water_drag_coeff * Ao)
        )
        f = np.sqrt(k) / (1 + np.sqrt(k))

        # environment
        if not vertical_profile:
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
        else:
            uprof = self.get_profile_masked("x_sea_water_velocity")
            vprof = self.get_profile_masked("y_sea_water_velocity")

            z = self.environment_profiles["z"]
            thickness = -(z[1:] - z[:-1]).reshape((-1, 1))
            mask = uprof.mask

            uprof_mean_inter = (uprof[1:] + uprof[:-1]) / 2
            vprof_mean_inter = (vprof[1:] + vprof[:-1]) / 2

            mask = mask[:-1]
            rows, cols = np.where(mask == False)
            rev_rows = rows[::-1]
            rev_cols = cols[::-1]
            unique_cols, rev_last_false_indices = np.unique(rev_cols, return_index=True)
            last_false_indices = len(rows) - 1 - rev_last_false_indices
            mask[rows[last_false_indices], cols[last_false_indices]] = True
            thickness_reshaped = np.tile(thickness, (1, mask.shape[1]))
            thickness_reshaped[mask] = np.nan

            umean = np.nansum(
                thickness_reshaped * uprof_mean_inter, axis=0
            ) / np.nansum(thickness_reshaped, axis=0)
            vmean = np.nansum(
                thickness_reshaped * vprof_mean_inter, axis=0
            ) / np.nansum(thickness_reshaped, axis=0)

            water_vel = np.array([umean, vmean])

        water_depth = self.environment.sea_floor_depth_below_sea_level
        wind_vel = np.array([self.environment.x_wind, self.environment.y_wind])
        wave_height = self.environment.sea_surface_wave_significant_height
        wave_direction = self.environment.sea_surface_wave_from_direction
        sea_ice_conc = self.environment.sea_ice_area_fraction
        sea_ice_thickness = self.environment.sea_ice_thickness
        sea_ice_vel = np.array(
            [self.environment.sea_ice_x_velocity, self.environment.sea_ice_y_velocity]
        )

        def dynamic(
            t,
            iceb_vel,
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
            iceb_length,
            mass,
        ):
            """Function required by solve_ivp. the t and iceb_vel parameters are required by solve_ivp. Don't delete them"""
            sum_force = (
                water_drag(t, iceb_vel, water_vel, Ao, rho_water, water_drag_coef)
                + wind_drag(t, iceb_vel, wind_vel, Aa, rho_air, wind_drag_coef)
                + int(wave_rad)
                * wave_radiation_force(
                    t,
                    iceb_vel,
                    rho_water,
                    wave_drag_coef,
                    g,
                    wave_height,
                    wave_direction,
                    iceb_length,
                )
            )
            sum_force = sum_force + sea_ice_force(
                t,
                iceb_vel,
                sea_ice_conc,
                sea_ice_thickness,
                sea_ice_vel,
                rhosi,
                self.elements.width,
                sum_force,
            )
            return 1 / mass * sum_force

        V0 = advect_iceberg_no_acc(
            f, water_vel, wind_vel
        )  # approximation of the solution of the dynamic equation for the iceberg velocity
        V0[:, sea_ice_conc >= 0.9] = sea_ice_vel[
            :, sea_ice_conc >= 0.9
        ]  # if this criteria, iceberg moves at sea ice speed
        V0 = V0.flatten()

        hwall = draft - water_depth
        grounded = hwall >= 0
        if any(grounded) and grounding:
            logger.info(
                f"Grounding : {len(hwall[hwall>0])}, hwall={np.round(hwall[hwall>0],3)}m"
            )
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
        Vx[grounded] = 0
        Vy[grounded] = 0
        self.update_positions(Vx, Vy)
        self.elements.iceb_x_velocity, self.elements.iceb_y_velocity = Vx, Vy

    def melt(self):
        """apply the different melting effects to the iceberg"""

        # loading the required variables :
        x_wind = self.environment.x_wind
        y_wind = self.environment.y_wind
        uoib = self.get_basal_env("x_sea_water_velocity")
        voib = self.get_basal_env("y_sea_water_velocity")
        T_profile = self.environment_profiles["sea_water_temperature"]
        S_profile = self.environment_profiles["sea_water_salinity"]
        Tn = self.get_basal_env("sea_water_temperature")
        Sn = self.get_basal_env("sea_water_salinity")
        ice_conc = self.environment.sea_ice_area_fraction

        # wave melting
        if self.choose_melting["wave"]:
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
        if self.choose_melting["lateral"]:
            self.elements.length, self.elements.width = mellat(
                self.elements.length,
                self.elements.width,
                T_profile,
                S_profile,
                self.time_step.total_seconds(),
            )
        # basal melting
        if self.choose_melting["basal"]:
            self.elements.draft, self.elements.sail = melbas(
                self.elements.draft,
                self.elements.sail,
                self.elements.length,
                Sn,
                Tn,
                uoib,
                voib,
                self.elements.iceb_x_velocity,
                self.elements.iceb_y_velocity,
                self.time_step.total_seconds(),
            )
        # deactivate elements too small : less than 1 meter
        self.deactivate_elements(self.elements.draft < 1, "Iceberg melted")
        self.deactivate_elements(self.elements.length < 1, "Iceberg melted")
        self.deactivate_elements(self.elements.width < 1, "Iceberg melted")
        self.deactivate_elements(self.elements.sail < 1, "Iceberg melted")

    def roll_over(self, rho_water):
        """Check the stability criteria of the iceberg and roll it over its smaller side if it's not satisfied. taken from Keghouche et al. 2009 with a correction on the stability criteria taken from Wagner et al. 2017

        Args:
            rho_water : sea water volumic mass [kg/m3]
        """
        L = self.elements.length
        W = self.elements.width
        H = self.elements.draft + self.elements.sail
        alpha = rho_iceb / rho_water
        crit = np.sqrt(6 * alpha * (1 - alpha))
        W, L = np.min([L, W], axis=0), np.max([L, W], axis=0)
        mask = (W / H) < crit
        if any(mask):
            logger.info(f"Rolling over : {np.sum(mask)} icebergs")
            nL, nW, nH = (
                np.max([L[mask], H[mask]], axis=0),
                np.min([L[mask], H[mask]], axis=0),
                W[mask],
            )
            L[mask], W[mask], H[mask] = nL, nW, nH
        depthib = H * alpha
        sailib = H - depthib
        self.elements.length = L
        self.elements.width = W
        self.elements.sail = sailib
        self.elements.draft = depthib

    # TODO check if prepare run is useless
    def prepare_run(self):
        self.profiles_depth = self.elements_scheduled.draft.max()
        logger.info(f"Max Icebergs draft is : {self.profiles_depth}")

    def update(self):
        """Update positions and properties of particles"""
        T = self.environment.sea_water_temperature
        S = self.environment.sea_water_salinity
        rho_water = PhysicsMethods.sea_water_density(T, S)
        self.roll_over(rho_water)
        if self.melting:
            self.melt()
        self.advect_iceberg(
            rho_water,
            self.add_stokes_drift,
            self.wave_rad,
            self.grounding,
            self.vertical_profile,
        )
