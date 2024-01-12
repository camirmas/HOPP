import pandas as pd
import openmdao.api as om

from hopp import ROOT_DIR
from hopp.utilities import load_yaml
from hopp.simulation.technologies.sites import SiteInfo, flatirons_site
from ev.optimization.components import HOPPComponent


DEFAULT_SOLAR_RESOURCE_FILE = ROOT_DIR.parent / "resource_files" / "solar" / "35.2018863_-101.945027_psmv3_60_2012.csv"
DEFAULT_WIND_RESOURCE_FILE = ROOT_DIR.parent / "resource_files" / "wind" / "35.2018863_-101.945027_windtoolkit_2012_60min_80m_100m.srw"
DEFAULT_PRICE_FILE = ROOT_DIR.parent / "resource_files" / "grid" / "pricing-data-2015-IronMtn-002_factors.csv"
EV_PATH = ROOT_DIR.parent / "ev"
EV_LOAD = pd.read_csv(EV_PATH / "data" / "boulder_demand_evi.csv", header=None)


def create_site() -> SiteInfo:
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


def run(case):
    print("running case: ", case)
    # set up site
    site = create_site()
    hopp_config = load_yaml(EV_PATH / "inputs" / "ev-load-following-battery.yaml")
    hopp_config["site"] = site

    # create model
    model = om.Group()
    prob = om.Problem(model)
    prob.driver = om.pyOptSparseDriver(optimizer="IPOPT")
    prob.driver.opt_settings["tol"] = 1e-6

    # bounds from optimization config
    battery_capacity_kw = case["battery_capacity_kw"]
    battery_capacity_kwh = case["battery_capacity_kwh"]
    threshold_kw = case["threshold_kw"]
    missed_allowed = case["missed_allowed"]

    hopp_config["technologies"]["battery"]["system_capacity_kw"] = battery_capacity_kw
    hopp_config["technologies"]["battery"]["system_capacity_kwh"] = battery_capacity_kwh
    hopp_config["config"]["dispatch_options"]["load_threshold_kw"] = threshold_kw

    # add components
    model.add_subsystem("HOPP", HOPPComponent(config=hopp_config, verbose=True), promotes=["*"])

    # model defaults

    # model.set_input_defaults("battery_capacity_kwh", capacity_lower)
    model.set_input_defaults("pv_rating_kw", threshold_kw)
    model.set_input_defaults("wind_rating_kw", threshold_kw)

    # add design vars
    # model.add_design_var("battery_capacity_kwh", lower = capacity_lower, units="kW*h")
    model.add_design_var("pv_rating_kw", lower=100, upper=1e6, units="kW")
    model.add_design_var("wind_rating_kw", lower=100, upper=1e6, units="kW")

    # add objective
    prob.model.add_objective("lcoe_real")    
    
    # add constraints

    ## avg missed peak load allowed
    prob.model.add_constraint("avg_missed_peak_load", upper=missed_allowed)

    # set up and run
    prob.setup()

    # recorder = om.SqliteRecorder(ROOT_DIR.parent / "ev" / "outputs" / "cases.sql")
    # prob.add_recorder(recorder)

    prob.run_driver()

    res = {
        "case": case,
        "technologies": {
            "pv": {
                "system_capacity_kw": prob.get_val("pv_rating_kw")[0],
            },
            "wind": {
                "turbine_rating_kw": prob.get_val("wind_rating_kw")[0],
            },
            "battery": {
                "system_capacity_kw": battery_capacity_kw,
                "system_capacity_kwh": battery_capacity_kwh
            }
        },
        "lcoe_real": prob.get_val("lcoe_real")[0],
        "avg_missed_peak_load": prob.get_val("avg_missed_peak_load")[0]
    }
    
    return res


if __name__ == "__main__":
    case = {
        "battery_capacity_kw": 600,
        "battery_capacity_kwh": 3000,
        "threshold_kw": 500,
        "missed_allowed": 30,
    }

    run(case)