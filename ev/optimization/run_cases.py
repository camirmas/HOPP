import os
from typing import List, Optional
import json

from ev.optimization.run_ev_opt import run

dir = os.path.dirname(os.path.realpath(__file__))
outputs_dir = dir + "/../outputs"


def load_case_results(dirname: Optional[str] = None) -> List[dict]:
    cases = []

    d = outputs_dir

    if dirname is not None:
        d = f"{outputs_dir}/{dirname}"

    for fname in os.listdir(d):
        with open(os.path.join(d, fname), "r") as f:
            cases.append(json.load(f))

    return cases


def get_best_case(dirname: Optional[str] = None) -> dict:
    """
    Gets the best case from the sweep results based on min LCOE.
    
    Returns:
        dict: The case with the best LCOE.
    """
    cases = load_case_results(dirname)

    best_case = cases[0]

    for case in cases:
        lcoe = case["lcoe_real"]

        if lcoe < best_case["lcoe_real"]:
            best_case = case

    return best_case


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

    for c in range(3, 11):
        cases[f"{c}_battery_hrs"] = {
            "threshold_kw": 500,
            "peak_req": .95,
            "battery_hrs": c
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

        fname = f"{dir}/../outputs/{name}.json"
        with open(fname, "w") as f:
            f.write(json.dumps(res))

        res["outfile"] = fname

        results.append(res)

    return results
        

if __name__ == "__main__":
    cases = create_cases()
    run_cases(cases)
