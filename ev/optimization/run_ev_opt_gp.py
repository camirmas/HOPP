from bayes_opt import BayesianOptimization
import pandas as pd

from hopp.simulation.technologies.sites import SiteInfo, flatirons_site
from hopp import ROOT_DIR
from hopp.utilities import load_yaml
from hopp.simulation import HoppInterface

DEFAULT_SOLAR_RESOURCE_FILE = ROOT_DIR.parent / "resource_files" / "solar" / "35.2018863_-101.945027_psmv3_60_2012.csv"
DEFAULT_WIND_RESOURCE_FILE = ROOT_DIR.parent / "resource_files" / "wind" / "35.2018863_-101.945027_windtoolkit_2012_60min_80m_100m.srw"
DEFAULT_PRICE_FILE = ROOT_DIR.parent / "resource_files" / "grid" / "pricing-data-2015-IronMtn-002_factors.csv"
EV_PATH = ROOT_DIR.parent / "ev"

def run_hopp(pv_capacity_kw, battery_capacity_kw, battery_capacity_kwh):
    ev_load = pd.read_csv(EV_PATH / "data" / "boulder_demand_evi.csv", header=None)

    site = SiteInfo(
            flatirons_site,
            solar_resource_file=str(DEFAULT_SOLAR_RESOURCE_FILE),
            wind_resource_file=str(DEFAULT_WIND_RESOURCE_FILE),
            grid_resource_file=str(DEFAULT_PRICE_FILE),
            desired_schedule=(ev_load.iloc[:, 0] / 1000).values,
            solar=True,
            wind=True,
            wave=False
        )
    
    hopp_config = load_yaml(EV_PATH / "inputs" / "ev-load-following-battery.yaml")
    # set SiteInfo instance
    hopp_config["site"] = site

    hopp_config["technologies"]["pv"]["system_capacity_kw"] = pv_capacity_kw
    hopp_config["technologies"]["battery"]["system_capacity_kw"] = battery_capacity_kw
    hopp_config["technologies"]["battery"]["system_capacity_kwh"] = battery_capacity_kwh

    hi = HoppInterface(hopp_config)
    hi.simulate()

    # missed_load_perc = hi.system.grid.missed_load_percentage * 100

    # if missed_load_perc > 20:
    #     return -10

    obj = hi.system.lcoe_real.hybrid/100.0 # convert from cents/kWh to USD/kWh

    return -obj

def run():
    pbounds = {
        "pv_capacity_kw": (1e2, 1e6),
        "battery_capacity_kw": (1e2, 1e6),
        "battery_capacity_kwh": (1e2, 1e6),
    }

    optimizer = BayesianOptimization(
        f = run_hopp,
        pbounds = pbounds,
    )

    optimizer.probe({
        "pv_capacity_kw": 2000,
        "battery_capacity_kw": 500,
        "battery_capacity_kwh": 2000,
    })

    optimizer.maximize(
        n_iter=100
    )

if __name__ == "__main__":
    run()