"""
Variable type management and unified variable creation for REINS.
"""

from enum import Enum

from neuromancer.constraint import Variable

import neuromancer as nm


class VarType(Enum):
    """
    Variable type enumeration.
    """
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    BINARY = "binary"

    def __repr__(self):
        return f"VarType.{self.name}"


def _build_var_types(num_vars, integer_indices=None, binary_indices=None):
    """Build a VarType list from num_vars and index lists."""
    # Check for overlap between integer and binary indices
    overlap = set(integer_indices or []) & set(binary_indices or [])
    if overlap:
        raise ValueError(
            f"Indices {overlap} appear in both integer_indices and "
            f"binary_indices. Each index must have exactly one type."
        )

    # Initialize all as continuous as default
    types = [VarType.CONTINUOUS] * num_vars

    # Set integer types based on indices
    for i in (integer_indices or []):
        if not 0 <= i < num_vars:
            raise ValueError(
                f"Integer index {i} out of range [0, {num_vars})"
            )
        types[i] = VarType.INTEGER

    # Set binary types based on indices
    for i in (binary_indices or []):
        if not 0 <= i < num_vars:
            raise ValueError(
                f"Binary index {i} out of range [0, {num_vars})"
            )
        types[i] = VarType.BINARY

    return types


def _resolve_var_types(num_vars, integer_indices, binary_indices, var_types):
    """Dispatch to explicit list or index-based construction."""
    if var_types is not None:
        if integer_indices is not None or binary_indices is not None:
            raise ValueError(
                "Cannot specify both var_types and indices-based parameters. "
                "Choose one approach: either pass var_types OR use indices."
            )
        # Single VarType -> broadcast to num_vars
        if isinstance(var_types, VarType):
            if num_vars is None:
                num_vars = 1
            return [var_types] * num_vars
        # List of VarTypes -> use directly
        return list(var_types)

    # Build from indices
    if num_vars is None:
        raise ValueError(
            "num_vars is required when using integer_indices or binary_indices."
        )
    return _build_var_types(num_vars, integer_indices, binary_indices)


class TypeVariable(Variable):
    """
    Typed variable for REINS mixed-integer optimization.

    Inherits from neuromancer Variable. Holds variable type metadata
    and provides ``.variable`` / ``.relaxed`` accessors for use in
    computation graphs.

    Note: Use ``.variable`` or ``.relaxed`` (plain neuromancer Variables)
    when building computation graph expressions. Do not use TypeVariable
    directly in arithmetic — neuromancer's ``__eq__`` is overridden here
    to prevent graph construction conflicts.

    Args:
        key: Variable name (must not end with '_rel').
        num_vars: Total number of variables.
        integer_indices: Indices of integer variables.
        binary_indices: Indices of binary variables.
        var_types: Single VarType (broadcast to all vars) or list of VarType.
            Mutually exclusive with indices-based parameters.
    """

    def __init__(self, key, num_vars=None,
                 integer_indices=None,
                 binary_indices=None,
                 var_types=None):
        # Validate key
        if key.endswith("_rel"):
            raise ValueError(
                f"Variable key '{key}' cannot end with '_rel' "
                f"(reserved for relaxed variables)."
            )

        # Initialize neuromancer Variable
        super().__init__(key=key)

        # Resolve type metadata
        types = _resolve_var_types(num_vars, integer_indices,
                                   binary_indices, var_types)

        # Store metadata
        self._var_types = types
        self._num_vars = len(types)
        self._integer_indices = [i for i, vt in enumerate(types)
                                 if vt == VarType.INTEGER]
        self._binary_indices = [i for i, vt in enumerate(types)
                                if vt == VarType.BINARY]
        self._continuous_indices = [i for i, vt in enumerate(types)
                                    if vt == VarType.CONTINUOUS]

        # Separate neuromancer Variables for computation graphs
        self._variable = nm.variable(key)
        self._relaxed = nm.variable(key + "_rel")

    # Override __eq__/__hash__ to prevent neuromancer's Constraint-returning
    # __eq__ from breaking make_graph's "not in" membership checks.
    # TypeVariable should not be used directly in computation graphs;
    # use .variable or .relaxed instead.
    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)

    @property
    def var_types(self):
        """List of VarType for each variable dimension."""
        return self._var_types

    @property
    def num_vars(self):
        """Total number of variable dimensions."""
        return self._num_vars

    @property
    def integer_indices(self):
        """Indices of integer-typed dimensions."""
        return self._integer_indices

    @property
    def binary_indices(self):
        """Indices of binary-typed dimensions."""
        return self._binary_indices

    @property
    def continuous_indices(self):
        """Indices of continuous-typed dimensions."""
        return self._continuous_indices

    @property
    def variable(self):
        """Neuromancer Variable for computation graphs."""
        return self._variable

    @property
    def relaxed(self):
        """Relaxed neuromancer Variable (key + '_rel')."""
        return self._relaxed

    def __repr__(self):
        return (f"TypeVariable(key='{self.key}', num_vars={self._num_vars}, "
                f"var_types={self._var_types})")
