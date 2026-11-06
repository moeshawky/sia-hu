"""Model Headroom Analyzer — assesses training headroom for MoE and dense models.

This package implements 7 diagnostic signals that measure training headroom:
loss saturation, growth factor bounds, effective rank, gradient-to-weight
ratio, dead neuron fraction, expert load balance, and specialization collapse.
The last two are expert-only: on dense models they are marked single_expert
and scored as a neutral 0.5 while staying in the composite denominator.

Dependencies: torch, transformers, numpy.
Provides: HeadroomReport with overall score and actionable recommendations.
"""
