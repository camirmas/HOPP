import os
from typing import List
import json

import pandas as pd

from ev.optimization.run_ev_opt import run

def create_cases() -> dict:
    """
    Creates a dictionary of different EV (Electric Vehicle) optimization cases.

    Each case contains parameters:
    - 'threshold_kw': The kW threshold for the case.
    - 'peak_req': Peak requirement, set at 0.95 for all cases.
    - 'battery_hrs': A dictionary specifying the lower and upper bounds for battery hours.

    Returns:
        dict: A dictionary where each key is a string in the format '{value}kw_threshold',
        and the value is a dictionary of case parameters.
    """
    cases = {}

    for c in [200, 400, 600, 800]:
        cases[f"{c}kw_threshold"] = {
            "threshold_kw": c,
            "peak_req": .95,
            "battery_hrs": {
                "lower": 5,
                "upper": 10
            }
        }
    
    return cases

def run_cases(cases: dict) -> List[dict]:
    """
    Runs optimization cases and writes the results to JSON files.

    Args:
        cases (dict): A dictionary of cases to run, where each case is a dictionary of parameters.

    Returns:
        list: A list of dicts containing results from each case.
    """
    results = []

    for name, case in cases.items():
        res = run(case)

        dir = os.path.dirname(os.path.realpath(__file__))
        fname = f"{dir}/../outputs/{name}.json"
        with open(fname, "w") as f:
            f.write(json.dumps(res))

        res["outfile"] = fname

        results.append(res)

    return results
        

if __name__ == "__main__":
    cases = create_cases()
    run_cases(cases)
