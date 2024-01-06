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
        
        self.add_input("battery_capacity_kw", units="kW")
        self.add_input("battery_capacity_kwh", units="kW*h")
        self.add_input("pv_rating_kw", units="kW")
        # self.add_input("wind_rating_kw", units="kW")

        self.add_output("missed_load_perc", units="percent")
        self.add_output("lcoe_real", units="USD/(kW*h)")

    def compute(self, inputs, outputs):
        config = self.options["config"]

        config["technologies"]["pv"]["system_capacity_kw"] = inputs["pv_rating_kw"][0]
        config["technologies"]["battery"]["system_capacity_kw"] = inputs["battery_capacity_kw"][0]
        config["technologies"]["battery"]["system_capacity_kwh"] = inputs["battery_capacity_kwh"][0]
        # config["technologies"]["wind"]["turbine_rating_kw"] = inputs["wind_rating_kw"][0]

        hi = HoppInterface(config)

        # run simulation
        hi.simulate(self.options["project_life"])

        if self.options["verbose"]:
            print(f"pv: {inputs['pv_rating_kw']}")
            # print(f"wind: {inputs['wind_rating_kw']}")
            print("battery (kW/kWh)", inputs["battery_capacity_kw"], inputs["battery_capacity_kwh"])
            print(f"missed load %: {hi.system.grid.missed_load_percentage * 100}")
            print(f"LCOE: {hi.system.lcoe_real.hybrid/100.0}\n")

        # get result
        outputs["missed_load_perc"] = hi.system.grid.missed_load_percentage * 100
        outputs["lcoe_real"] = hi.system.lcoe_real.hybrid/100.0 # convert from cents/kWh to USD/kWh

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd', form="forward")


class BatteryResilienceComponent(om.ExplicitComponent):
    """Used for constraints related to battery capacity and resilience."""

    def initialize(self):
        self.options.declare("config", recordable=False, types=dict)
        self.options.declare("verbose", types=bool)

    def setup(self):
        self.add_input("battery_capacity_kw", units="kW")
        self.add_input("battery_capacity_kwh", units="kW*h")

        self.add_output("battery_hours", units="h")

    def compute(self, inputs, outputs):
        outputs["battery_hours"] = inputs["battery_capacity_kwh"] / inputs["battery_capacity_kw"]

        if self.options["verbose"]:
            print(f"battery hours: {outputs['battery_hours']}\n")

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd', form="forward")