"""
Gradient-based feasibility projection.
"""

import torch


class GradientProjection:
    """
    Gradient-based feasibility projection.

    Args:
        rounding_components: List of rounding modules.
        constraints: List of Constraint objects.
        target_keys: Keys for relaxed variables to project.
        num_steps: Maximum projection iterations.
        step_size: Initial step size for gradient descent.
        decay: Step size decay factor per iteration.
        tolerance: Stop if max violation < tolerance.
    """

    def __init__(self, rounding_components, constraints, target_keys,
                 num_steps=1000, step_size=0.01, decay=1.0,
                 tolerance=1e-6):
        self.rounding_components = rounding_components
        self.constraints = constraints
        self.target_keys = target_keys
        self.num_steps = num_steps
        self.step_size = step_size
        self.decay = decay
        self.tolerance = tolerance

    def __call__(self, data):
        """
        Project relaxed variables towards feasibility.

        Args:
            data: Dictionary containing variable tensors.

        Returns:
            Updated dictionary with projected and rounded solution.
        """
        # Clone and enable grad for all target variables
        xs = {k: data[k].clone().requires_grad_(True) for k in self.target_keys}
        batch_size = next(iter(xs.values())).shape[0]
        d = 1.0

        # Build temp data once, update in-place each iteration
        temp_data = {**data}
        for _ in range(self.num_steps):
            temp_data.update(xs)

            # Round through components
            for comp in self.rounding_components:
                temp_data.update(comp(temp_data))

            # Compute total violation from all constraints at once
            viols = []
            for con in self.constraints:
                out = con(temp_data)
                viol_key = con.output_keys[2]
                viols.append(out[viol_key].reshape(batch_size, -1).sum(dim=1))
            if not viols:
                break
            total_viol = torch.stack(viols).sum(dim=0) if len(viols) > 1 else viols[0]

            # Check convergence
            if total_viol.max().item() < self.tolerance:
                break

            # Gradient step on all target variables
            grads = torch.autograd.grad(
                total_viol.sum(), list(xs.values()),
                allow_unused=True,
            )
            xs = {
                k: (xs[k] - d * self.step_size * g).detach().requires_grad_(True)
                if g is not None else xs[k]
                for k, g in zip(self.target_keys, grads)
            }
            d = self.decay * d

        # Final update
        for k in self.target_keys:
            data[k] = xs[k].detach()

        # Final round
        for comp in self.rounding_components:
            data.update(comp(data))

        return data
