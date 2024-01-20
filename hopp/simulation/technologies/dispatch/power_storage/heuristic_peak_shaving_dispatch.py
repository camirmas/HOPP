from typing import Optional, List, TYPE_CHECKING

import pyomo.environ as pyomo
import PySAM.BatteryStateful as BatteryModel
import PySAM.Singleowner as Singleowner

if TYPE_CHECKING:
    from hopp.simulation.technologies.dispatch.hybrid_dispatch_options import HybridDispatchOptions

from hopp.simulation.technologies.dispatch.power_storage.simple_battery_dispatch_heuristic import SimpleBatteryDispatchHeuristic


class HeuristicPeakShavingDispatch(SimpleBatteryDispatchHeuristic):
    """
    Operates the battery based on heuristic rules to meet a threshold based on
    power available from power generation profiles and power demand profile.

    Currently, enforces available generation and grid limit assuming no battery
    charging from grid.
    """
    def __init__(self,
                 pyomo_model: pyomo.ConcreteModel,
                 index_set: pyomo.Set,
                 system_model: BatteryModel.BatteryStateful,
                 financial_model: Singleowner.Singleowner,
                 fixed_dispatch: Optional[List] = None,
                 block_set_name: str = 'heuristic_load_following_battery',
                 dispatch_options: Optional["HybridDispatchOptions"] = None):
        """

        Args:
            fixed_dispatch: list of normalized values [-1, 1] (Charging (-), Discharging (+))
        """
        if dispatch_options is None:
            raise ValueError("Dispatch options must be set for this dispatch method")

        super().__init__(
            pyomo_model,
            index_set,
            system_model,
            financial_model,
            fixed_dispatch,
            block_set_name,
            dispatch_options
        )

    def _set_power_fraction_limits(self, gen: list, grid_limit: list):
        """
        Set battery charge and discharge power fraction limits based on
        available generation and grid capacity, respectively.

        Args:
            gen: generation Blocks
            grid_limit: grid capacity

        NOTE: This method assumes that battery cannot be charged by the grid.
        """
        for t in self.blocks.index_set():
            self.max_charge_fraction[t] = self.enforce_power_fraction_simple_bounds(gen[t] / self.maximum_power)
            self.max_discharge_fraction[t] = self.enforce_power_fraction_simple_bounds((grid_limit[t] - gen[t])
                                                                                       / self.maximum_power)

    def set_fixed_dispatch(self, gen: list, grid_limit: list, goal_power: list):
        """
        Sets charge and discharge power of battery dispatch using fixed_dispatch attribute and enforces available
        generation and grid limits.
        """
        self.check_gen_grid_limit(gen, grid_limit)
        self._set_power_fraction_limits(gen, grid_limit)
        self._heuristic_method(gen, goal_power)
        self._fix_dispatch_model_variables()

    def _heuristic_method(self, gen, goal_power):
        """ 
        Enforces battery power fraction limits and sets _fixed_dispatch attribute.

        Sets the _fixed_dispatch based on goal_power, gen (power generation profile), and
        peak shaving threshold.
        """
        threshold_mw = self.options.load_threshold_kw / 1000

        for t in self.blocks.index_set():
            # peak loads: use battery, note that this could mean charging if we
            #   have high generation
            if goal_power[t] > threshold_mw:
                fd = (goal_power[t] - threshold_mw - gen[t]) / self.maximum_power
            
            # non-peak: charge with non-dispatchable generation
            else:
                fd = - gen[t] / self.maximum_power
            
            if fd > 0.0:    # Discharging
                if fd > self.max_discharge_fraction[t]:
                    fd = self.max_discharge_fraction[t]
            elif fd < 0.0:  # Charging
                if -fd > self.max_charge_fraction[t]:
                    fd = -self.max_charge_fraction[t]
            self._fixed_dispatch[t] = fd        