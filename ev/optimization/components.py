import openmdao.api as om
import numpy as np

from hopp.simulation import HoppInterface


class HOPPComponent(om.ExplicitComponent):

    def initialize(self):
        self.options.declare("config", recordable=False, types=dict)
        self.options.declare("project_life", default=25, types=int)
        self.options.declare("verbose", types=bool)

    def setup(self):
        config = self.options["config"]

        # self.add_input("battery_capacity_kwh", units="kW*h")
        self.add_input("pv_rating_kw", units="kW")
        self.add_input("wind_rating_kw", units="kW")

        self.add_output("missed_load_perc", units="percent")
        self.add_output("avg_missed_peak_load", units="kW")
        self.add_output("lcoe_real", units="USD/(kW*h)")
        self.add_output("curtailed", units="kW")

    def compute(self, inputs, outputs):
        config = self.options["config"]

        config["technologies"]["pv"]["system_capacity_kw"] = inputs["pv_rating_kw"][0]
        # config["technologies"]["battery"]["system_capacity_kw"] = inputs["battery_capacity_kw"][0]
        # config["technologies"]["battery"]["system_capacity_kwh"] = inputs["battery_capacity_kwh"][0]
        config["technologies"]["wind"]["turbine_rating_kw"] = inputs["wind_rating_kw"][0]

        hi = HoppInterface(config)

        # run simulation
        hi.simulate(self.options["project_life"])

        # get result
        outputs["missed_load_perc"] = hi.system.grid.missed_load_percentage * 100
        outputs["avg_missed_peak_load"] = np.mean(hi.system.grid.missed_peak_load)
        outputs["lcoe_real"] = hi.system.lcoe_real.hybrid/100.0 # convert from cents/kWh to USD/kWh

        if self.options["verbose"]:
            print(f"pv: {inputs['pv_rating_kw']}")
            print(f"wind: {inputs['wind_rating_kw']}")
            # print("battery (kWh)", inputs["battery_capacity_kwh"])
            print(f"avg missed peak load (kW): {outputs['avg_missed_peak_load']}")
            print(f"LCOE: {outputs['lcoe_real']}\n")

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd', form="forward")