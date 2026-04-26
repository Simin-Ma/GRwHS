from __future__ import annotations

from typing import Sequence

from .schemas import (
    DEFAULT_ABLATION_VARIANTS,
    DEFAULT_STANDARD_METHODS,
    MechanismSettingSpec,
)


def _build_ablation_setting(
    p0_true: int,
    *,
    ablation_variants: Sequence[str],
    dense_ablation: bool,
) -> MechanismSettingSpec:
    include_in_paper_table = int(p0_true) == 5
    return MechanismSettingSpec(
        setting_id=f"m4_ablation_p0_{int(p0_true):02d}",
        setting_label=f"M4 Ablation p0={int(p0_true)}",
        experiment_id="M4",
        experiment_label="Mechanism Ablation",
        experiment_kind="ablation",
        line_id="exp4",
        line_label="Exp4",
        scientific_question="Which part of GR-RHS is responsible for the mechanism advantage?",
        primary_metric="kappa_gap",
        group_sizes=(10, 10, 10, 10, 10),
        n_train=100,
        n_test=30,
        rho_within=0.8,
        rho_between=0.2,
        sigma2=1.0,
        total_active_coeff=int(p0_true),
        suite="dense_ablation" if dense_ablation else "mechanism",
        role=(
            "ablation and mechanism attribution"
            if include_in_paper_table
            else "ablation diagnostic under dense support"
        ),
        notes=(
            "Random mixed strong/weak coefficient support with oracle-reference tau diagnostics."
            if include_in_paper_table
            else "Dense-support ablation diagnostic retained as an optional stress slice, not the default mechanism suite."
        ),
        include_in_paper_table=include_in_paper_table,
        methods=tuple(str(item) for item in ablation_variants),
    )


def build_dense_ablation_settings(
    *,
    ablation_variants: Sequence[str] = DEFAULT_ABLATION_VARIANTS,
) -> tuple[MechanismSettingSpec, ...]:
    return tuple(
        _build_ablation_setting(
            p0_true,
            ablation_variants=ablation_variants,
            dense_ablation=True,
        )
        for p0_true in (15, 30)
    )


def build_mechanism_suite(
    *,
    standard_methods: Sequence[str] = DEFAULT_STANDARD_METHODS,
    ablation_variants: Sequence[str] = DEFAULT_ABLATION_VARIANTS,
    include_dense_ablation: bool = False,
) -> tuple[MechanismSettingSpec, ...]:
    standard = tuple(str(item) for item in standard_methods)
    ablation = tuple(str(item) for item in ablation_variants)
    out: list[MechanismSettingSpec] = []

    out.append(
        MechanismSettingSpec(
            setting_id="m1_group_separation",
            setting_label="M1 Group Separation",
            experiment_id="M1",
            experiment_label="Group Separation",
            experiment_kind="group_separation",
            line_id="ga_v2_a",
            line_label="GA-V2-A",
            scientific_question="Does GR-RHS learn stronger signal-group vs null-group separation?",
            primary_metric="kappa_gap",
            group_sizes=(10, 10, 10, 10, 10),
            n_train=100,
            n_test=30,
            rho_within=0.8,
            rho_between=0.2,
            sigma2=1.0,
            mu=(0.0, 0.0, 1.5, 4.0, 10.0),
            role="cleanest direct mechanism check",
            notes="Heterogeneous group-level signal strengths with fixed sigma2.",
            methods=standard,
        )
    )

    for within_group_pattern in ("mixed_decoy", "concentrated"):
        for rho_within in (0.8, 0.9):
            out.append(
                MechanismSettingSpec(
                    setting_id=f"m2_{within_group_pattern}_rw{int(round(100 * rho_within)):03d}",
                    setting_label=f"M2 {within_group_pattern} rw={rho_within:.2f}",
                    experiment_id="M2",
                    experiment_label="Correlation Stress Under Structural Ambiguity",
                    experiment_kind="correlation_stress",
                    line_id="ga_v2_c",
                    line_label="GA-V2-C",
                    scientific_question="Does group-aware shrinkage matter more under correlated ambiguity and decoy null groups?",
                    primary_metric="mse_overall",
                    group_sizes=(10, 10, 10, 10, 10),
                    active_groups=(0, 1),
                    n_train=100,
                    n_test=30,
                    rho_within=float(rho_within),
                    rho_between=0.2,
                    target_snr=1.0,
                    within_group_pattern=str(within_group_pattern),
                    role="ambiguity stress test",
                    notes="mixed_decoy is the headline design; concentrated is the contrast case.",
                    methods=standard,
                )
            )

    for complexity_pattern in ("few_groups", "many_groups"):
        for within_group_pattern in ("concentrated", "distributed"):
            out.append(
                MechanismSettingSpec(
                    setting_id=f"m3_{complexity_pattern}_{within_group_pattern}",
                    setting_label=f"M3 {complexity_pattern} {within_group_pattern}",
                    experiment_id="M3",
                    experiment_label="Complexity Unit / Scope Condition",
                    experiment_kind="complexity_mismatch",
                    line_id="ga_v2_b",
                    line_label="GA-V2-B",
                    scientific_question="Does GR-RHS respond to group complexity rather than only coefficient count?",
                    primary_metric="kappa_gap",
                    group_sizes=(10, 10, 10, 10, 10),
                    n_train=100,
                    n_test=30,
                    rho_within=0.8,
                    rho_between=0.2,
                    target_snr=1.0,
                    within_group_pattern=str(within_group_pattern),
                    complexity_pattern=str(complexity_pattern),
                    total_active_coeff=10,
                    role="scope condition rather than universal win",
                    notes="Hold total active coefficients fixed while changing group allocation.",
                    methods=standard,
                )
            )

    out.append(
        _build_ablation_setting(
            5,
            ablation_variants=ablation,
            dense_ablation=False,
        )
    )
    if include_dense_ablation:
        out.extend(build_dense_ablation_settings(ablation_variants=ablation))

    return tuple(out)


def get_setting_by_id(
    setting_id: str,
    *,
    standard_methods: Sequence[str] = DEFAULT_STANDARD_METHODS,
    ablation_variants: Sequence[str] = DEFAULT_ABLATION_VARIANTS,
    include_dense_ablation: bool = True,
) -> MechanismSettingSpec:
    target = str(setting_id)
    for setting in build_mechanism_suite(
        standard_methods=standard_methods,
        ablation_variants=ablation_variants,
        include_dense_ablation=include_dense_ablation,
    ):
        if setting.setting_id == target:
            return setting
    raise KeyError(f"Unknown mechanism setting id: {setting_id!r}")
