"""
Unit tests for PenaltyLoss.
"""

import pytest
import torch
import neuromancer as nm
from neuromancer.loss import PenaltyLoss as _NMPenaltyLoss

from reins.loss import PenaltyLoss


@pytest.fixture(autouse=True)
def seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)


def _make_constraint(var_key, upper_bound, weight=1.0, name="con"):
    """Create an inequality constraint: x <= upper_bound."""
    x = nm.variable(var_key)
    con = weight * (x <= upper_bound)
    con.name = name
    return con


def _make_eq_constraint(var_key, target, weight=1.0, name="eq_con"):
    """Create an equality constraint: x == target."""
    x = nm.variable(var_key)
    # Use Variable for target to avoid F.l1_loss scalar-vs-tensor broadcasting warning
    t = nm.variable(target) if isinstance(target, str) else target
    con = weight * (x == t)
    con.name = name
    return con


class TestPenaltyLossType:
    """Test PenaltyLoss class hierarchy."""

    def test_is_subclass(self):
        assert issubclass(PenaltyLoss, _NMPenaltyLoss)

    def test_export(self):
        import reins
        assert hasattr(reins, "PenaltyLoss")
        assert "PenaltyLoss" in reins.__all__
        assert reins.PenaltyLoss is PenaltyLoss


class TestPenaltyLossSumReduction:
    """Test sum-reduction over constraint dimensions."""

    def test_single_constraint_sum_reduction(self):
        """Violation should be summed over dims, then averaged over batch."""
        con = _make_constraint("x", upper_bound=0.0, weight=1.0)
        loss = PenaltyLoss(objectives=[], constraints=[con])
        # x shape: (2, 3), all positive -> all violate x <= 0
        data = {"x": torch.tensor([[1.0, 2.0, 3.0],
                                    [4.0, 5.0, 6.0]])}
        output = loss.calculate_constraints(data)
        # Violations: [1, 2, 3] and [4, 5, 6]
        # Sum per sample: 6, 15
        # Mean over batch: (6 + 15) / 2 = 10.5
        expected = 10.5
        assert torch.isclose(output["penalty_loss"], torch.tensor(expected))

    def test_sum_vs_mean_reduction(self):
        """REINS PenaltyLoss (sum) should differ from neuromancer (mean)."""
        con = _make_constraint("x", upper_bound=0.0, weight=1.0)
        reins_loss = PenaltyLoss(objectives=[], constraints=[con])
        nm_loss = _NMPenaltyLoss(objectives=[], constraints=[con])
        # Multi-dim constraint
        data = {"x": torch.tensor([[1.0, 2.0, 3.0]])}
        reins_out = reins_loss.calculate_constraints(data)
        nm_out = nm_loss.calculate_constraints(data)
        # REINS: sum([1,2,3]) = 6.0
        # neuromancer: mean([1,2,3]) = 2.0
        assert reins_out["penalty_loss"].item() == pytest.approx(6.0)
        assert nm_out["penalty_loss"].item() == pytest.approx(2.0)

    def test_no_violation(self):
        """No violation should produce zero penalty loss."""
        con = _make_constraint("x", upper_bound=10.0)
        loss = PenaltyLoss(objectives=[], constraints=[con])
        data = {"x": torch.tensor([[1.0, 2.0, 3.0]])}
        output = loss.calculate_constraints(data)
        assert output["penalty_loss"].item() == pytest.approx(0.0)

    def test_weight_scaling(self):
        """Constraint weight should scale the penalty."""
        con = _make_constraint("x", upper_bound=0.0, weight=3.0)
        loss = PenaltyLoss(objectives=[], constraints=[con])
        data = {"x": torch.tensor([[1.0, 2.0]])}
        output = loss.calculate_constraints(data)
        # Violation sum: 1+2 = 3, weight 3 -> 9.0
        assert output["penalty_loss"].item() == pytest.approx(9.0)


class TestPenaltyLossOutputKeys:
    """Test output dictionary keys and shapes."""

    def test_output_keys_present(self):
        """Output should contain violation and value tensors."""
        con = _make_constraint("x", upper_bound=5.0)
        loss = PenaltyLoss(objectives=[], constraints=[con])
        data = {"x": torch.tensor([[1.0, 2.0]])}
        output = loss.calculate_constraints(data)
        assert "penalty_loss" in output
        assert "C_violations" in output
        assert "C_values" in output
        assert "C_eq_violations" in output
        assert "C_ineq_violations" in output

    def test_violation_shape(self):
        """C_violations shape should be (batch, total_constraint_dims)."""
        con = _make_constraint("x", upper_bound=5.0)
        loss = PenaltyLoss(objectives=[], constraints=[con])
        data = {"x": torch.tensor([[1.0, 2.0, 3.0],
                                    [4.0, 5.0, 6.0]])}
        output = loss.calculate_constraints(data)
        assert output["C_violations"].shape == (2, 3)
        assert output["C_values"].shape == (2, 3)

    def test_ineq_flags(self):
        """Inequality constraints should appear in C_ineq_violations."""
        con = _make_constraint("x", upper_bound=5.0)
        loss = PenaltyLoss(objectives=[], constraints=[con])
        data = {"x": torch.tensor([[1.0, 2.0]])}
        output = loss.calculate_constraints(data)
        assert output["C_ineq_violations"].shape[1] == 2
        assert output["C_eq_violations"].shape[1] == 0

    def test_eq_flags(self):
        """Equality constraints should appear in C_eq_violations."""
        con = _make_eq_constraint("x", target="t")
        loss = PenaltyLoss(objectives=[], constraints=[con])
        data = {"x": torch.tensor([[1.0, 2.0]]), "t": torch.zeros(1, 2)}
        output = loss.calculate_constraints(data)
        assert output["C_eq_violations"].shape[1] == 2
        assert output["C_ineq_violations"].shape[1] == 0


class TestPenaltyLossMultipleConstraints:
    """Test with multiple constraints."""

    def test_two_constraints(self):
        """Penalties from multiple constraints should be summed."""
        con1 = _make_constraint("x", upper_bound=0.0, name="con1")
        con2 = _make_constraint("x", upper_bound=0.0, name="con2")
        loss = PenaltyLoss(objectives=[], constraints=[con1, con2])
        data = {"x": torch.tensor([[1.0, 2.0]])}
        output = loss.calculate_constraints(data)
        # Each constraint: sum([1,2]) = 3.0, total: 6.0
        assert output["penalty_loss"].item() == pytest.approx(6.0)

    def test_no_constraints(self):
        """No constraints should produce zero penalty."""
        loss = PenaltyLoss(objectives=[], constraints=[])
        data = {"x": torch.tensor([[1.0, 2.0]])}
        output = loss.calculate_constraints(data)
        assert float(output["penalty_loss"]) == pytest.approx(0.0)
        assert "C_violations" not in output

    def test_no_constraints_returns_tensor(self):
        """penalty_loss should be a tensor even with no constraints."""
        loss = PenaltyLoss(objectives=[], constraints=[])
        data = {"x": torch.tensor([[1.0]])}
        output = loss.calculate_constraints(data)
        assert isinstance(output["penalty_loss"], torch.Tensor)


class TestPenaltyLossNumerical:
    """Verify exact numerical penalty loss values."""

    def test_partial_violation(self):
        """Only violating dims should contribute to penalty."""
        con = _make_constraint("x", upper_bound=2.0, weight=1.0)
        loss = PenaltyLoss(objectives=[], constraints=[con])
        # x = [1.0, 3.0, 2.0]: only dim 1 violates (3.0 > 2.0, violation = 1.0)
        data = {"x": torch.tensor([[1.0, 3.0, 2.0]])}
        output = loss.calculate_constraints(data)
        assert output["penalty_loss"].item() == pytest.approx(1.0)

    def test_violation_values_exact(self):
        """Verify C_violations contains per-element violation magnitudes."""
        con = _make_constraint("x", upper_bound=1.0, weight=1.0)
        loss = PenaltyLoss(objectives=[], constraints=[con])
        data = {"x": torch.tensor([[0.5, 1.5, 3.0]])}
        output = loss.calculate_constraints(data)
        violations = output["C_violations"]
        # dim 0: 0.5 <= 1.0, no violation -> 0.0
        # dim 1: 1.5 > 1.0, violation -> 0.5
        # dim 2: 3.0 > 1.0, violation -> 2.0
        assert violations[0, 0].item() == pytest.approx(0.0)
        assert violations[0, 1].item() == pytest.approx(0.5)
        assert violations[0, 2].item() == pytest.approx(2.0)
        # penalty_loss = sum([0, 0.5, 2.0]) = 2.5
        assert output["penalty_loss"].item() == pytest.approx(2.5)

    def test_equality_constraint_numerical(self):
        """Equality constraint violation should be |x - target|."""
        con = _make_eq_constraint("x", target="t", weight=1.0)
        loss = PenaltyLoss(objectives=[], constraints=[con])
        data = {
            "x": torch.tensor([[1.0, 3.0]]),
            "t": torch.tensor([[2.0, 1.0]]),
        }
        output = loss.calculate_constraints(data)
        # |1-2| + |3-1| = 1 + 2 = 3.0
        assert output["penalty_loss"].item() == pytest.approx(3.0)

    def test_mixed_ineq_eq_constraints(self):
        """Mixed inequality and equality constraints summed correctly."""
        ineq = _make_constraint("x", upper_bound=0.0, weight=1.0, name="ineq")
        eq = _make_eq_constraint("x", target="t", weight=1.0, name="eq")
        loss = PenaltyLoss(objectives=[], constraints=[ineq, eq])
        data = {
            "x": torch.tensor([[2.0, 1.0]]),
            "t": torch.tensor([[0.0, 0.0]]),
        }
        output = loss.calculate_constraints(data)
        # ineq: x <= 0: violations = [2, 1], sum = 3.0
        # eq: |x - t| = [2, 1], sum = 3.0
        # total: 6.0
        assert output["penalty_loss"].item() == pytest.approx(6.0)
        assert output["C_ineq_violations"].shape[1] == 2
        assert output["C_eq_violations"].shape[1] == 2

    def test_batch_mean_over_samples(self):
        """Penalty should be sum-per-sample then mean-over-batch."""
        con = _make_constraint("x", upper_bound=0.0, weight=1.0)
        loss = PenaltyLoss(objectives=[], constraints=[con])
        data = {"x": torch.tensor([[1.0, 1.0],
                                    [3.0, 3.0],
                                    [5.0, 5.0]])}
        output = loss.calculate_constraints(data)
        # Sample 1: sum([1, 1]) = 2
        # Sample 2: sum([3, 3]) = 6
        # Sample 3: sum([5, 5]) = 10
        # Mean: (2 + 6 + 10) / 3 = 6.0
        assert output["penalty_loss"].item() == pytest.approx(6.0)

    def test_different_weights(self):
        """Different constraint weights should scale independently."""
        con1 = _make_constraint("x", upper_bound=0.0, weight=2.0, name="c1")
        con2 = _make_constraint("x", upper_bound=0.0, weight=5.0, name="c2")
        loss = PenaltyLoss(objectives=[], constraints=[con1, con2])
        data = {"x": torch.tensor([[1.0]])}
        output = loss.calculate_constraints(data)
        # con1: 2.0 * 1.0 = 2.0, con2: 5.0 * 1.0 = 5.0, total: 7.0
        assert output["penalty_loss"].item() == pytest.approx(7.0)

    def test_lower_bound_constraint(self):
        """x >= lower_bound: violation when x < bound."""
        x = nm.variable("x")
        con = 1.0 * (x >= 2.0)
        con.name = "lb_con"
        loss = PenaltyLoss(objectives=[], constraints=[con])
        data = {"x": torch.tensor([[0.5, 1.5, 3.0]])}
        output = loss.calculate_constraints(data)
        # dim 0: relu(2.0 - 0.5) = 1.5
        # dim 1: relu(2.0 - 1.5) = 0.5
        # dim 2: relu(2.0 - 3.0) = 0.0
        # sum = 2.0
        assert output["penalty_loss"].item() == pytest.approx(2.0)

    def test_c_violations_exact_content(self):
        """C_violations should contain per-element relu violation magnitudes."""
        con = _make_constraint("x", upper_bound=2.0, weight=1.0)
        loss = PenaltyLoss(objectives=[], constraints=[con])
        data = {"x": torch.tensor([[1.0, 2.0, 4.0]])}
        output = loss.calculate_constraints(data)
        viols = output["C_violations"]
        # dim 0: 1.0 <= 2.0 -> 0.0
        # dim 1: 2.0 <= 2.0 -> 0.0
        # dim 2: 4.0 > 2.0 -> 2.0
        assert viols[0, 0].item() == pytest.approx(0.0)
        assert viols[0, 1].item() == pytest.approx(0.0)
        assert viols[0, 2].item() == pytest.approx(2.0)

    def test_equality_zero_violation(self):
        """x == target with exact match should produce zero violation."""
        con = _make_eq_constraint("x", target="t", weight=1.0)
        loss = PenaltyLoss(objectives=[], constraints=[con])
        data = {
            "x": torch.tensor([[3.0, 5.0]]),
            "t": torch.tensor([[3.0, 5.0]]),
        }
        output = loss.calculate_constraints(data)
        assert output["penalty_loss"].item() == pytest.approx(0.0)
