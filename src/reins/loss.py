"""
Loss functions for REINS.
"""

import math

import torch
import numpy as np
from neuromancer.loss import PenaltyLoss as _NMPenaltyLoss


class PenaltyLoss(_NMPenaltyLoss):
    """
    PenaltyLoss with sum-reduction over constraint dimensions.

    Unlike neuromancer's default mean-reduction, violations are summed
    over non-batch dimensions before averaging over the batch.
    """

    def calculate_constraints(self, input_dict):
        loss = 0.0
        output_dict = {}
        C_values = []
        C_violations = []
        eq_flags = []
        for c in self.constraints:
            output = c(input_dict)
            output_dict = {**output_dict, **output}
            cvalue = output[c.output_keys[1]]
            cviolation = output[c.output_keys[2]]
            # sum over constraint dims, mean over batch
            flat = cviolation.reshape(cviolation.shape[0], -1)
            loss += c.weight * flat.sum(dim=1).mean()
            nr_constr = math.prod(cvalue.shape[1:])
            eq_flags += nr_constr * [str(c.comparator) == 'eq']
            C_values.append(cvalue.reshape(cvalue.shape[0], -1))
            C_violations.append(flat)
        if self.constraints:
            equalities_flags = np.array(eq_flags)
            C_violations = torch.cat(C_violations, dim=-1)
            C_values = torch.cat(C_values, dim=-1)
            output_dict['C_violations'] = C_violations
            output_dict['C_values'] = C_values
            output_dict['C_eq_violations'] = C_violations[:, equalities_flags]
            output_dict['C_ineq_violations'] = C_violations[:, ~equalities_flags]
            output_dict['C_eq_values'] = C_values[:, equalities_flags]
            output_dict['C_ineq_values'] = C_values[:, ~equalities_flags]
        output_dict['penalty_loss'] = loss
        return output_dict
