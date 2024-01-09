import pandas as pd
import openmdao.api as om

from hopp import ROOT_DIR
from hopp.utilities import load_yaml
from hopp.simulation.technologies.sites import SiteInfo, flatirons_site
from .components import HOPPComponent


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
    # set up site
    site = create_site()
    hopp_config = load_yaml(EV_PATH / "inputs" / "ev-load-following-battery.yaml")
    hopp_config["site"] = site

    # create model
    model = om.Group()
    prob = om.Problem(model)
    prob.driver = om.pyOptSparseDriver(optimizer="IPOPT")
    prob.driver.opt_settings["tol"] = 1e-3

    # bounds from optimization config
    threshold_kw = case["threshold_kw"]
    peak_req = case["peak_req"]

    capacity_lower = threshold_kw * case["battery_hrs"]

    hopp_config["technologies"]["battery"]["system_capacity_kw"] = threshold_kw
    hopp_config["technologies"]["battery"]["system_capacity_kwh"] = capacity_lower
    hopp_config["config"]["dispatch_options"]["load_threshold_kw"] = threshold_kw

    # add components
    model.add_subsystem("HOPP", HOPPComponent(config=hopp_config, verbose=True), promotes=["*"])

    # model defaults
    battery_init_c = hopp_config["technologies"]["battery"]["system_capacity_kwh"]
    pv_init = hopp_config["technologies"]["pv"]["system_capacity_kw"]
    wind_init = hopp_config["technologies"]["wind"]["turbine_rating_kw"]

    model.set_input_defaults("battery_capacity_kwh", capacity_lower)
    model.set_input_defaults("pv_rating_kw", pv_init)
    model.set_input_defaults("wind_rating_kw", wind_init)

    # add design vars
    model.add_design_var("battery_capacity_kwh", lower = capacity_lower, units="kW*h")
    model.add_design_var("pv_rating_kw", lower=100, upper=1e6, units="kW")
    model.add_design_var("wind_rating_kw", lower=100, upper=1e6, units="kW")

    # add objective
    prob.model.add_objective("lcoe_real", ref=1e-2)    
    
    # add constraints

    ## avg missed peak load <= some % of peak threshold
    prob.model.add_constraint("avg_missed_peak_load", upper=(1 - peak_req) * threshold_kw)

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
                "system_capacity_kw": threshold_kw,
                "system_capacity_kwh": prob.get_val("battery_capacity_kwh")[0],
            }
        },
        "lcoe_real": prob.get_val("lcoe_real")[0],
    }
    
    return res


if __name__ == "__main__":
    case = {
        "threshold_kw": 500,
        "peak_req": .95,
        "battery_hrs": {
            "lower": 5,
            "upper": 10
        }
    }

    run(case)