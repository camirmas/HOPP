import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import openmdao.api as om

from hopp import ROOT_DIR
from hopp.simulation import HoppInterface
from hopp.utilities import load_yaml
from hopp.simulation.technologies.sites import SiteInfo, flatirons_site
from hopp.tools.dispatch.plot_tools import (
    plot_battery_output, plot_battery_dispatch_error, plot_generation_profile
)
from components import BatteryResilienceComponent, HOPPComponent


DEFAULT_SOLAR_RESOURCE_FILE = ROOT_DIR.parent / "resource_files" / "solar" / "35.2018863_-101.945027_psmv3_60_2012.csv"
DEFAULT_WIND_RESOURCE_FILE = ROOT_DIR.parent / "resource_files" / "wind" / "35.2018863_-101.945027_windtoolkit_2012_60min_80m_100m.srw"
DEFAULT_PRICE_FILE = ROOT_DIR.parent / "resource_files" / "grid" / "pricing-data-2015-IronMtn-002_factors.csv"
EV_PATH = ROOT_DIR.parent / "ev"
EV_LOAD = pd.read_csv(EV_PATH / "data" / "boulder_demand_evi.csv", header=None)


def create_site():
    site = SiteInfo(
            flatirons_site,
            solar_resource_file=str(DEFAULT_SOLAR_RESOURCE_FILE),
            wind_resource_file=str(DEFAULT_WIND_RESOURCE_FILE),
            grid_resource_file=str(DEFAULT_PRICE_FILE),
            desired_schedule=(EV_LOAD.iloc[:, 0] / 1000).values,
            solar=True,
            wind=True,
            wave=False
        )

    return site


def run():
    # set up site
    site = create_site()
    hopp_config = load_yaml(EV_PATH / "inputs" / "ev-load-following-battery.yaml")
    hopp_config["site"] = site

    # create model
    model = om.Group()
    prob = om.Problem(model)
    prob.driver = om.pyOptSparseDriver(optimizer="IPOPT")
    prob.driver.opt_settings["tol"] = 1e-3

    # add components
    model.add_subsystem("HOPP", HOPPComponent(config=hopp_config, verbose=True), promotes=["*"])
    model.add_subsystem("con_battery", BatteryResilienceComponent(verbose=False), promotes=["*"])

    model.set_input_defaults("battery_capacity_kw", hopp_config["technologies"]["battery"]["system_capacity_kw"])
    model.set_input_defaults("battery_capacity_kwh", hopp_config["technologies"]["battery"]["system_capacity_kwh"])
    model.set_input_defaults("pv_rating_kw", hopp_config["technologies"]["pv"]["system_capacity_kw"])
    model.set_input_defaults("wind_rating_kw", hopp_config["technologies"]["wind"]["turbine_rating_kw"])

    # add design vars
    model.add_design_var("battery_capacity_kwh", lower=0, upper=1e9, units="kW*h")
    model.add_design_var("battery_capacity_kw", lower=0, upper=1e9, units="kW")
    model.add_design_var("pv_rating_kw", lower=0, upper=1e9, units="kW")
    model.add_design_var("wind_rating_kw", lower=500, upper=1e9, units="kW")

    # objective
    prob.model.add_objective("lcoe_real", ref=1e-2)    
    
    # constraints
    prob.model.add_constraint("battery_hours", lower=9, upper=10)
    prob.model.add_constraint("battery_capacity_kw", lower=np.median(EV_LOAD))
    prob.model.add_constraint("missed_load_perc", upper=25)

    # set up and run
    prob.setup()

    recorder = om.SqliteRecorder(ROOT_DIR.parent / "ev" / "outputs" / "cases.sql")
    prob.add_recorder(recorder)

    prob.run_driver()


def plot_outputs(hi: HoppInterface):
    hybrid_plant = hi.system
    plot_battery_dispatch_error(hybrid_plant)
    plot_battery_output(hybrid_plant)
    plot_generation_profile(hybrid_plant)

    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(8760)
    y = hybrid_plant.battery.outputs.dispatch_P

    periods = 24*7
    dates = pd.date_range(start="2022-01-01", periods=periods, freq="H")

    ax.plot(dates, y[:periods], linewidth=.6, label="Battery")
    ax.plot(dates, hybrid_plant.site.desired_schedule[:periods], linewidth=.6, label="EV Demand")

    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45)
            
    ax.set_xlabel("Hour")
    ax.set_ylabel("Power (MW)")
    ax.set_xmargin(0)

    ax.legend()


if __name__ == "__main__":
    run()