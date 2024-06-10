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
    V = np.array([no_acc_model_x, no_acc_model_y])
    if not np.isfinite(V).all():
        pass
    return V


def sea_ice_force(
    t, iceb_vel, sea_ice_conc, sea_ice_thickness, sea_ice_vel, rhosi, wib, sum_force
):
    csi = 1  # sea ice coefficient of resistance
    ice_x, ice_y = sea_ice_vel
    x_vel, y_vel = iceb_vel[0], iceb_vel[1]
    diff_vel = np.sqrt((ice_x - x_vel) ** 2 + (ice_y - y_vel) ** 2)
    force_x, force_y = sum_force
    F_ice_x = np.zeros_like(x_vel)
    F_ice_y = np.zeros_like(y_vel)
    F_ice_x = 0.5 * (rhosi * csi * sea_ice_thickness * wib) * diff_vel * (ice_x - x_vel)
    F_ice_y = 0.5 * (rhosi * csi * sea_ice_thickness * wib) * diff_vel * (ice_y - y_vel)
    F_ice_x[sea_ice_conc <= 0.15] = 0
    F_ice_y[sea_ice_conc <= 0.15] = 0
    F_ice_x[sea_ice_conc >= 0.9] = -force_x[sea_ice_conc >= 0.9]
    F_ice_y[sea_ice_conc >= 0.9] = -force_y[sea_ice_conc >= 0.9]
    return np.array([F_ice_x, F_ice_y])


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
    Vwe = Vsst * 0.5 * (1 + np.cos(np.pi * conc**3)) / 86400  # melting in m/s to Check

    # length lost only on one side
    new_lib = np.zeros_like(lib)
    new_wib = np.zeros_like(wib)
    new_lib[lib != 0] = lib[lib != 0] - Vwe[lib != 0] * dt
    new_wib[lib != 0] = wib[lib != 0] / lib[lib != 0] * new_lib[lib != 0]
    new_lib[new_lib < 0] = 0
    new_wib[new_wib < 0] = 0
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
    deltaT = np.concatenate([2.78 * deltaT, 0.47 * deltaT**2], axis=0)
    # deltaT = np.concatenate([deltaT, deltaT**2], axis=0) Old

    # coefs = np.concatenate(
    #     [np.ones_like(tempib) * 2.78, np.ones_like(tempib) * 0.47], axis=0 Old
    # )
    # sumVb = np.diag(np.dot(deltaT.T, coefs)) Old
    sumVb = np.nansum(deltaT, axis=0)

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
    new_lib[new_lib < 0] = 0
    new_wib[new_wib < 0] = 0
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
    absv = np.sqrt(((uoib - uib) ** 2 + (voib - vib) ** 2))
    TfS = -0.036 - 0.0499 * salnib - 0.000112 * salnib**2
    Tfp = TfS * 2.71828 ** (-0.19 * (tempib - TfS))
    deltat = tempib - Tfp

    Vf = 0.58 * absv**0.8 * deltat / (lib**0.2)  # / 86400
    Vf = Vf / 86400  # convert in m/s

    # Update the depth
    new_depthib = np.zeros_like(depthib)
    new_depthib[depthib != 0] = (
        abs(depthib[depthib != 0]) - Vf[depthib != 0] * dt
    )  # melt the iceberg base
    new_depthib[depthib < 0] = 0

    return new_depthib, sailib


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
            variable (_type_): _description_
        """
        draft = self.elements.draft
        profile = self.environment_profiles[variable]
        z = self.environment_profiles["z"]
        mask = draft[:, np.newaxis] < -z
        mask = mask.T
        mask[np.argmax(mask, axis=0), np.arange(mask.shape[1])] = False
        return np.ma.masked_array(profile, mask, fill_value=np.nan)

    def get_basal_env(self, variable):
        profile = self.get_profile_masked(variable)
        last = np.argmin(np.logical_not(profile.mask), axis=0) - 1
        return profile[last, np.arange(profile.shape[1])]

    def seed_ensemble(
        self,
        width: tuple,
        height: tuple,
        Ca: tuple,
        Co: tuple,
        numbers: tuple = (10, 10, 10, 10, 1),
        random=True,
        drag_coeff_distribution="normal",
        **kwargs,
    ):
        """_summary_

        Args:
            width (tuple): (mean,std)
            height (tuple): (mean,std)
            Ca (tuple): (min,max,mean,std)
            Co (tuple): (min,max,mean,std)
            numbers (tuple, optional): number of distinct member per param. Defaults to (10,10,10,10,10).

        Returns:
            _type_: _description_
        """
        w_mean, w_std = width
        h_mean, h_std = height
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

        if random:
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
        else:
            Ens_w = np.abs(np.random.normal(w_mean, w_std, numbers[0]))
            Ens_h = np.abs(np.random.normal(h_mean, h_std, numbers[1]))
            if drag_coeff_distribution == "normal":
                Ens_Ca = np.abs(np.random.normal(Ca_mean, Ca_std, numbers[2]))
                Ens_Co = np.abs(np.random.normal(Co_mean, Co_std, numbers[3]))
            elif drag_coeff_distribution == "uniform":
                Ens_Ca = np.abs(
                    np.random.uniform(Ca_mean - Ca_std, Ca_mean + Ca_std, numbers[2])
                )
                Ens_Co = np.abs(
                    np.random.uniform(Co_mean - Co_std, Co_mean + Co_std, numbers[3])
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

        ### Roll over stability : we decide to keep it random whitout correction. Uncomment to add the correction before running the model
        # Ens_h[Ens_h > Ens_w / crit] = (
        #     Ens_h[Ens_h > Ens_w / crit] / 2
        # )  # Roll over stability
        if not random:
            Ens_tot = np.array(list(itertools.product(Ens_w, Ens_h, Ens_Ca, Ens_Co)))
            logger.info("seeding ensemble ...")
            for member in Ens_tot:
                w, h, ca, co = member
                l = w
                draft = h * alpha
                sail = h - draft
                self.seed_elements(
                    **kwargs,
                    number=numbers[-1],
                    width=w,
                    length=l,
                    draft=draft,
                    sail=sail,
                    water_drag_coeff=co,
                    wind_drag_coeff=ca,
                )
        else:
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
            # for member in sampled_array:
            #     w, h, ca, co = member
            #     l = w
            #     draft = h * alpha
            #     sail = h - draft
            #     self.seed_elements(
            #         **kwargs,
            #         width=w,
            #         length=l,
            #         draft=draft,
            #         sail=sail,
            #         water_drag_coeff=co,
            #         wind_drag_coeff=ca,
            #     )
        return 0

    # Configuration
    def __init__(
        self,
        with_stokes_drift: bool = True,
        wave_rad: bool = True,
        grounding: bool = False,
        water_profile: bool = False,
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
        self.with_stokes_drift = with_stokes_drift
        self.grounding = grounding
        self.water_profile = water_profile
        self.melting = (
            melting  # boolean parameter to decide wether or not the iceberg melt
        )
        self.choose_melting = (
            choose_melting  # boolean dictionnary to decide how the iceberg melt
        )

    def advect_iceberg(
        self, stokes_drift=True, wave_rad=True, grounding=False, water_profile=False
    ):

        # Constants
        rho_water = 1027
        rho_air = 1.293
        rho_iceb = 900
        rhosi = 917
        wave_drag_coef = 0.3
        g = 9.81
        draft = self.elements.draft
        height = self.elements.sail + draft
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
        if not water_profile:  # TODO change in vert_profile
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
            # Make the average current that covers the Iceberg's draft
            # uprof = self.environment_profiles[
            #     "x_sea_water_velocity"
            # ]
            # vprof = self.environment_profiles["y_sea_water_velocity"]
            uprof = self.get_profile_masked("x_sea_water_velocity")
            vprof = self.get_profile_masked("y_sea_water_velocity")
            z = self.environment_profiles["z"]
            thickness = -(z[1:] - z[:-1]).reshape((-1, 1))
            mask = uprof.mask
            # print(z[np.logical_not(mask)], draft)
            uprof_mean_inter = (uprof[1:] + uprof[:-1]) / 2
            vprof_mean_inter = (vprof[1:] + vprof[:-1]) / 2
            mask = mask[:-1]
            # uprof_mean_inter[mask] = np.nan
            # vprof_mean_inter[mask] = np.nan
            thickness_reshaped = np.tile(thickness, (1, mask.shape[1]))
            thickness_reshaped[mask] = np.nan
            # print(np.nansum(thickness_reshaped * uprof_mean_inter, axis=0))
            umean = np.nansum(
                thickness_reshaped * uprof_mean_inter, axis=0
            ) / np.nansum(thickness_reshaped, axis=0)
            vmean = np.nansum(
                thickness_reshaped * vprof_mean_inter, axis=0
            ) / np.nansum(thickness_reshaped, axis=0)
            # print(umean)
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
            sum_force = sum_force + sea_ice_force(
                t,
                V,
                sea_ice_conc,
                sea_ice_thickness,
                sea_ice_vel,
                rhosi,
                self.elements.width,
                sum_force,
            )
            return 1 / mass * sum_force

        V0 = advect_iceberg_no_acc(f, water_vel, wind_vel)
        V0[:, sea_ice_conc >= 0.9] = sea_ice_vel[
            :, sea_ice_conc >= 0.9
        ]  # if this criteria, iceberg moves at sea ice speed
        V0 = V0.flatten()

        hwall = draft - water_depth
        grounded = hwall >= 0  # Check
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
        L = self.elements.length
        W = self.elements.width
        H = self.elements.draft + self.elements.sail
        rho_iceb = 900
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
            self.with_stokes_drift, self.wave_rad, self.grounding, self.water_profile
        )
