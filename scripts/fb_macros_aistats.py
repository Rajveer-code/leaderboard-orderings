"""G4.4 -- LaTeX macros for every number the manuscript may state, plus the claim ledger.

Each macro is read from a committed result file; none is typed here. The ledger records, for
every macro, its value and the exact file and key it came from, so a reviewer (or the author)
can trace any number in the paper back to the artifact that produced it.

Run: python scripts/fb_macros_aistats.py
Out: manuscript/tmlr/macros.tex, CLAIM_LEDGER_AISTATS.md
"""
import gzip
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "flipbudget"
TEX = ROOT / "manuscript" / "tmlr" / "macros.tex"
LEDGER = ROOT / "CLAIM_LEDGER_AISTATS.md"


def fmt(v):
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}".replace(",", "{,}")
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def long_date(iso):
    """2025-01-31 -> 31 January 2025, for dates that tie a count to its moment."""
    from datetime import date
    d = date.fromisoformat(iso)
    return f"{d.day} {d.strftime('%B')} {d.year}"


def labeller_macros(lab):
    """Reliability of the 400-item audit's labels: agreement on the reader verdict, the string-level
    agreement recorded with the overlaps, and the certified counts when a labeller is swapped."""
    out, src = [], "labeller_agreement.json"
    for tag, k in (("Two", "two"), ("Three", "three")):
        y = lab["y_star"][k]
        zero, allc = y["by_cell"]["000"], y["by_cell"]["111"]
        out += [(f"Lab{tag}N", y["n"], src), (f"Lab{tag}Agree", y["agree"], src),
                (f"Lab{tag}Pct", f'{100 * y["agree"] / y["n"]:.1f}', src),
                (f"Lab{tag}Kappa", round(y["kappa"], 2), src),
                (f"Lab{tag}ZeroN", zero[1], src), (f"Lab{tag}ZeroAgree", zero[0], src),
                (f"Lab{tag}AllN", allc[1], src), (f"Lab{tag}AllDis", allc[1] - allc[0], src)]
    sl = lab["string_level"]
    for name, key in (("OneTwo", "l1_vs_l2"), ("OneThree", "l1_vs_l3"), ("TwoThree", "l2_vs_l3")):
        out += [(f"Str{name}Pct", f'{100 * sl[key]["raw_agreement"]:.1f}', "mathhard_human_margins.json: inter_rater"),
                (f"Str{name}Kappa", round(sl[key]["cohens_kappa"], 2), "mathhard_human_margins.json: inter_rater")]
    base = lab["certified"]["baseline"]
    for tag, k in (("Two", "replace_with_two"), ("Three", "replace_with_three")):
        c = lab["certified"][k]
        out += [(f"LabSwap{tag}EpsZero", c["eps0"], src), (f"LabSwap{tag}Prereg", c["prereg"], src)]
    out.append(("LabSwapMax", max(abs(lab["certified"][k]["prereg"] - base["prereg"])
                                  for k in ("replace_with_two", "replace_with_three")), src))
    return out


def projection_macros():
    """The census projection that decided D4, recorded before any census label (PROJECTED)."""
    proj = json.load(open(RES / "g3_census_projection.json", encoding="utf-8"))["scenarios"]["uniform"]
    src = "g3_census_projection.json: scenarios.uniform (PROJECTED)"
    return [("CensusProjectedMedian", int(proj["census"]["median"]), src),
            ("CensusProjectedLow", int(proj["census"]["p10"]), src),
            ("CensusProjectedHigh", int(proj["census"]["p90"]), src),
            ("RegisteredProjectedMedian", int(proj["registered"]["median"]), src)]


def post_label_macros(K3, eps0):
    """Results that exist only once human labels do. Nothing is emitted before then, so no
    sentence can cite them early; Section 5.5 stays PENDING until these are defined."""
    out = []
    v3_path = RES / "audit_v3_labels.json"
    if v3_path.exists() and "with_v3" in K3:
        v3 = json.load(open(v3_path, encoding="utf-8"))
        w = K3["with_v3"]
        src = "audit_v3_labels.json / certify_results.json: K3.with_v3"
        grid = {p["eps"]: p for p in w["eps_curve"]}
        rows = v3["falsification"]["sympy"]["rows"]
        out += [("VThreeNLabels", v3["n_labels"], src),
                ("VThreeExcludedBatches", len(v3["excluded_batches"]), src),
                ("VThreeScreenRows", len(rows), src),
                ("VThreeScreenRejected", sum(r["reject"] for r in rows), src),
                ("VThreeCertFixedEpsZero", w["fixed_board"]["certified_eps0"], src),
                ("VThreeCertFixedPrereg", w["fixed_board"]["certified_eps_preregistered"], src),
                ("VThreeCertPrereg", grid[eps0]["combined"], src)]
        d8 = v3.get("d8_extraction_restored")
        if d8 is not None and "d8_extraction_restored" in w:
            out += [("VThreeRestoredBatches", len(d8["restored_batches"]), src),
                    ("VThreeCertFixedPreregDEight", w["d8_extraction_restored"]["fixed_board"]
                     ["certified_eps_preregistered"], src),
                    ("VThreeScreenRejectedDEight", sum(r["reject"] for r in d8["falsification"]["sympy"]["rows"]),
                     src)]
        ov = v3.get("overlap", {})
        if ov.get("kappa") is not None:
            out += [("KappaOverlap", round(ov["kappa"], 2), src), ("KappaN", ov["n_common_scored"], src)]
    if "census" in K3:
        src = "certify_results.json: K3.census"
        g, d = K3["census"]["gate|sympy|as_incorrect"], K3["census"]["d8_extraction_restored|sympy|as_incorrect"]
        out += [("CensusDomains", g["completeness"]["domains"], src),
                ("CensusDomainsComplete", g["completeness"]["complete"], src),
                ("CensusLabelled", g["completeness"]["labelled"], src),
                ("CensusResponses", g["completeness"]["responses"], src),
                ("CensusCertFixedEpsZero", g["fixed_board"]["certified_eps0"], src),
                ("CensusCertFixedPrereg", g["fixed_board"]["certified_eps_preregistered"], src),
                ("CensusCertFixedPreregDEight", d["fixed_board"]["certified_eps_preregistered"], src),
                ("CensusLabelledDEight", d["completeness"]["labelled"], src)]
    return out


def main():
    cert = json.load(open(RES / "certify_results.json", encoding="utf-8"))
    prov = json.load(open(RES / "verifier_provenance.json", encoding="utf-8"))
    cov = json.load(open(RES / "coverage_table.json", encoding="utf-8"))
    amap = json.load(open(RES / "audit_k3_map.json", encoding="utf-8"))
    sim = json.load(open(RES / "certify_coverage_sim.json", encoding="utf-8"))
    front = json.load(open(RES / "g3_frontier.json", encoding="utf-8"))
    op = json.load(open(RES / "g3_operating_point.json", encoding="utf-8"))
    imp = json.load(open(RES / "g1_semantics_impact.json", encoding="utf-8"))
    tests = json.load(open(RES / "test_census.json", encoding="utf-8"))
    ms = json.load(open(ROOT / "results" / "analysis" / "model_subject_table.json", encoding="utf-8"))["summary"]
    bd = json.load(open(RES / "board_response_dates.json", encoding="utf-8"))
    gc = json.load(open(ROOT / "results" / "analysis" / "gold_consistency.json", encoding="utf-8"))
    gf = json.load(open(ROOT / "results" / "analysis" / "gold_ft.json", encoding="utf-8"))
    sens = cert["K3"]["sensitivity"]
    ext = json.load(open(RES / "external_sources.json", encoding="utf-8"))
    rp = json.load(open(RES / "roster_provenance.json", encoding="utf-8"))
    bpos = json.load(open(RES / "board_position.json", encoding="utf-8"))
    bshift = json.load(open(RES / "board_verifier_shift.json", encoding="utf-8"))
    brp = json.load(open(RES / "board_row_provenance.json", encoding="utf-8"))
    srd = brp["same_run_detail"]
    btime = json.load(open(RES / "board_timepoints.json", encoding="utf-8"))["revisions"]
    lab = json.load(open(RES / "labeller_agreement.json", encoding="utf-8"))
    tax = json.load(open(RES / "extraction_taxonomy.json", encoding="utf-8"))
    pert = json.load(open(RES / "perturbation.json", encoding="utf-8"))["credited_after_rewrite"]
    k4 = cert.get("K4", {})
    xfv = json.load(gzip.open(RES / "xfinder_verdicts.json.gz", "rt", encoding="utf-8"))["verdicts"]
    pct = lambda x: round(100 * x, 1)  # noqa: E731
    ac = tax["audited_cross"]
    m17 = amap["models17"]
    disc_dom = {}
    for it in amap["items"]:
        if it.get("k3_sympy") and it["k3_sympy"] not in ("000", "111"):
            disc_dom[(it["model"], it["k3_sympy"])] = disc_dom.get((it["model"], it["k3_sympy"]), 0) + 1
    HC = ROOT / "results" / "harness_census"
    prev = {r[0]: r for r in json.load(open(HC / "prevalence.json", encoding="utf-8"))["rows"]}
    light = json.load(open(HC / "census_lighteval.json", encoding="utf-8"))
    simple = json.load(open(HC / "simple_evals_census.json", encoding="utf-8"))
    valid = json.load(open(HC / "census_validation.json", encoding="utf-8"))["summary"]
    census = json.load(gzip.open(RES / "census_k3.json.gz", "rt", encoding="utf-8"))
    K3, eps0 = cert["K3"], cert["config"]["eps0_preregistered"]
    grid = {p["eps"]: p for p in K3["eps_curve"]}
    cells3 = amap["census_cells"]["k3_sympy"]["27"]
    rows = census["rows"]
    opt = next(p for p in front["points"] if p.get("is_operating_point"))
    worst_sim = max(s["audit"]["familywise_error"] for s in sim["scenarios"]
                    if s["eps_true"] >= 0 and s["eps_true"] == s["eps_analysis"])

    M = [  # (macro, value, source)
        ("NumModels", len(cov["models_kept"]), "coverage_table.json: models_kept"),
        ("NumItems", cert["config"]["n_items"], "certify_results.json: config.n_items"),
        ("NumOrderings", len(K3["eps0_pairs"]), "certify_results.json: K3.eps0_pairs"),
        ("NumVerifiers", len(K3["inputs"]["cells"][0]), "certify_results.json: K3.inputs.cells (cell width)"),
        ("NumCells", len(K3["inputs"]["cells"]), "certify_results.json: K3.inputs.cells"),
        ("NumRows", len(rows), "census_k3.json.gz: rows"),
        ("MissingRows", sum(v["n_missing"] for v in cov["per_model"].values()), "coverage_table.json: per_model"),
        ("CommonItems", cov["n_common_items"], "coverage_table.json: n_common_items"),
        # provenance
        ("EmReexecDisagree", prov["summary"]["items_em_vs_em_rec_disagree"], "verifier_provenance.json: summary"),
        ("EmReexecItems", prov["summary"]["items"], "verifier_provenance.json: summary.items"),
        ("EmAggregateMatch", prov["summary"]["subjects_em_equals_original_aggregate"], "verifier_provenance.json"),
        # only models whose results file was rescored in Feb 2025 can test the Math-Verify match
        ("MvAggregateMatch", sum(v["mv_matches_current_aggregate"] for m in prov["per_model"].values()
                                 if m["results_original_revision"] for v in m["per_subject"].values()),
         "verifier_provenance.json: per_model (rescored models only)"),
        ("NumSubjectsRescored", prov["summary"]["subjects"]
         - 7 * prov["summary"]["models_without_pre_rescoring_results"], "verifier_provenance.json: summary"),
        # cells and audit
        ("CellAgreeAll", cells3["111"], "audit_k3_map.json: census_cells.k3_sympy.27"),
        ("CellRejectAll", cells3["000"], "audit_k3_map.json: census_cells.k3_sympy.27"),
        ("RowsAnyDisagree", sum(len({r["em"], r["sb_sympy"], r["mv"]}) > 1 for r in rows), "census_k3.json.gz"),
        ("DisagreeSbEmSeventeen", amap["disagreements_17_sb_sympy_vs_em"], "audit_k3_map.json"),
        ("DisagreeSbEmSeventeenString", amap["disagreements_17_sb_string_vs_em"], "audit_k3_map.json"),
        ("AuditCorrectAgreeAll", K3["inputs"]["audit_pooled"]["111"]["x_correct"], "certify_results.json: K3.inputs"),
        ("AuditNAgreeAll", K3["inputs"]["audit_pooled"]["111"]["n"], "certify_results.json: K3.inputs"),
        ("AuditCorrectRejectAll", K3["inputs"]["audit_pooled"]["000"]["x_correct"], "certify_results.json: K3.inputs"),
        ("AuditNRejectAll", K3["inputs"]["audit_pooled"]["000"]["n"], "certify_results.json: K3.inputs"),
        # certification
        # fixed-board estimand (primary): these items, simultaneous over the roster, exact
        ("CertFixedEpsZero", K3["fixed_board"]["certified_eps0"], "certify_results.json: K3.fixed_board"),
        ("CertFixedPrereg", K3["fixed_board"]["certified_eps_preregistered"],
         "certify_results.json: K3.fixed_board"),
        ("CertFixedNoPooling", K3["fixed_board"]["certified_no_pooling"], "certify_results.json: K3.fixed_board"),
        ("CertFixedKTwoEpsZero", cert["K2"]["fixed_board"]["certified_eps0"], "certify_results.json: K2.fixed_board"),
        ("CertFixedKTwoPrereg", cert["K2"]["fixed_board"]["certified_eps_preregistered"],
         "certify_results.json: K2.fixed_board"),
        # rounded down to 3 decimals: a reported breakdown value may understate, never overstate
        ("BreakdownMedianFixed", math.floor(sorted(K3["fixed_board"]["breakdown_eps"].values())[
            len(K3["fixed_board"]["breakdown_eps"]) // 2] * 1000) / 1000,
         "certify_results.json: K3.fixed_board.breakdown_eps median, floored to 3 decimals"),
        ("FamilyLevelFixed", round(1 - cert["config"]["alpha_audit"], 4), "1 - config.alpha_audit"),
        # what the rank intervals mean for a reader of the board (width = highest minus lowest rank)
        ("RankWidthMedianEpsZero", sorted(hi - lo for lo, hi in K3["fixed_board"]["rank_intervals_eps0"].values())[
            len(K3["fixed_board"]["rank_intervals_eps0"]) // 2], "certify_results.json: fixed_board.rank_intervals_eps0"),
        ("RankWidthMedianPrereg", sorted(hi - lo for lo, hi in K3["fixed_board"]["rank_intervals_eps_preregistered"].values())[
            len(K3["fixed_board"]["rank_intervals_eps_preregistered"]) // 2],
         "certify_results.json: fixed_board.rank_intervals_eps_preregistered"),
        ("RanksResolvedPrereg", sum(hi == lo for lo, hi in K3["fixed_board"]["rank_intervals_eps_preregistered"].values()),
         "certify_results.json: fixed_board.rank_intervals_eps_preregistered"),
        ("SensDropFixed", sens["indeterminate_dropped"]["fixed_board"][1],
         "certify_results.json: K3.sensitivity.indeterminate_dropped.fixed_board[prereg]"),
        ("SensCpFixed", sens["clopper_pearson"]["fixed_board"][1],
         "certify_results.json: K3.sensitivity.clopper_pearson.fixed_board[prereg]"),
        ("SensMvFixed", sens["mv_reference"]["fixed_board"][1],
         "certify_results.json: K3.sensitivity.mv_reference.fixed_board[prereg]"),
        ("SensMvFixedEpsZero", sens["mv_reference"]["fixed_board"][0],
         "certify_results.json: K3.sensitivity.mv_reference.fixed_board[eps0]"),
        ("SensStringFixed", sens["string_semantics"]["fixed_board"][1],
         "certify_results.json: K3.sensitivity.string_semantics.fixed_board[prereg]"),
        ("SensSeventeenFixed", sens["models17"]["fixed_board"][1],
         "certify_results.json: K3.sensitivity.models17.fixed_board[prereg]"),
        # item-generalising estimand (secondary)
        ("CertEpsZero", grid[0.0]["combined"], "certify_results.json: K3.eps_curve[eps=0]"),
        ("CertEpsZeroSim", grid[0.0]["combined_simultaneous"], "certify_results.json: K3.eps_curve[eps=0]"),
        ("CertEpsZeroAudit", grid[0.0]["audit_only"], "certify_results.json: K3.eps_curve[eps=0]"),
        ("CertPrereg", grid[eps0]["combined"], f"certify_results.json: K3.eps_curve[eps={eps0}]"),
        ("CertPreregSim", grid[eps0]["combined_simultaneous"], f"certify_results.json: K3.eps_curve[eps={eps0}]"),
        ("CertNoPooling", K3["no_pooling"]["combined_resolved"], "certify_results.json: K3.no_pooling"),
        ("CeilingEm", K3["ceiling"]["em"], "certify_results.json: K3.ceiling.em"),
        ("CeilingMv", K3["ceiling"]["mv"], "certify_results.json: K3.ceiling.mv"),
        ("CertKTwoEpsZero", cert["K2"]["eps_curve"][0]["combined"], "certify_results.json: K2.eps_curve"),
        ("CertKTwoPrereg", {p["eps"]: p for p in cert["K2"]["eps_curve"]}[eps0]["combined"],
         "certify_results.json: K2.eps_curve"),
        ("EpsPrereg", eps0, "certify_results.json: config.eps0_preregistered"),
        ("EpsRefDerived", K3["eps_reference_a"]["eps_ref_a"], "certify_results.json: K3.eps_reference_a"),
        ("BreakdownMedian", sorted(K3["breakdown_eps"].values())[len(K3["breakdown_eps"]) // 2],
         "certify_results.json: K3.breakdown_eps (median)"),
        # the Windows defect, measured
        ("AnchorStringCombined", imp["string"]["as_incorrect"]["combined_resolved"], "g1_semantics_impact.json"),
        ("AnchorSympyCombined", imp["sympy"]["as_incorrect"]["combined_resolved"], "g1_semantics_impact.json"),
        ("AnchorStringAudit", imp["string"]["as_incorrect"]["audit_resolved"], "g1_semantics_impact.json"),
        ("AnchorSympyAudit", imp["sympy"]["as_incorrect"]["audit_resolved"], "g1_semantics_impact.json"),
        # coverage simulation
        ("SimReplicates", sim["n_rep"], "certify_coverage_sim.json: n_rep"),
        ("SimWorstFwerUnderH", worst_sim, "certify_coverage_sim.json: scenarios where H holds"),
        ("SimFwerMisspecified", next(s["audit"]["familywise_error"] for s in sim["scenarios"]
                                     if s["eps_true"] == 0.02), "certify_coverage_sim.json"),
        ("SimFwerPropensityKTwo", next(s["audit"]["familywise_error"] for s in sim["scenarios"]
                                       if s["K"] == 2 and s["eps_true"] < 0 and s["eps_analysis"] == 0),
         "certify_coverage_sim.json"),
        ("SimFwerPropensityKThree", next(s["audit"]["familywise_error"] for s in sim["scenarios"]
                                         if s["K"] == 3 and s["eps_true"] < 0 and s["eps_analysis"] == 0),
         "certify_coverage_sim.json: K=3, homogeneous propensities, analysed at eps=0"),
        # zero failures in n replicates: exact one-sided upper confidence bound 1 - (1 - level)^(1/n)
        ("SimZeroFwerConfPct", 95, "confidence level of SimZeroFwerUpper (design constant)"),
        # rounded up to 4 decimals: it is an upper bound
        ("SimZeroFwerUpper", math.ceil((1 - 0.05 ** (1 / sim["n_rep"])) * 10000) / 10000,
         "1 - 0.05^(1/n_rep), n_rep from certify_coverage_sim.json, ceiled to 4 decimals"),
        ("SimScenarios", len(sim["scenarios"]), "certify_coverage_sim.json: scenarios"),
        ("SimMcSeAudit", float(f'{sim["scenarios"][0]["audit"]["mc_se_at_nominal"]:.2g}'),  # two significant figures
         "certify_coverage_sim.json: audit.mc_se_at_nominal"),
        # levels, and the distribution-free sampling alternative (appendix C)
        ("AlphaAudit", cert["config"]["alpha_audit"], "certify_results.json: config.alpha_audit"),
        ("AlphaSample", cert["config"]["alpha_sampling"], "certify_results.json: config.alpha_sampling"),
        ("SidakTwoLevel", cert["K3"]["inputs"]["interval_level"],
         "certify_results.json: K3.inputs.interval_level"),
        ("CertHoeffding", cert["K3"]["sensitivity"]["sampling_hoeffding"]["combined"],
         "certify_results.json: K3.sensitivity.sampling_hoeffding.combined"),
        ("CertSamplingLegacy", cert["K3"]["sensitivity"]["sampling_legacy"]["combined"],
         "certify_results.json: K3.sensitivity.sampling_legacy.combined"),
        # board responses unchanged across the re-scoring (section 6, appendix G)
        ("BoardDateSampleN", bd["n_sampled"], "board_response_dates.json: n_sampled"),
        ("BoardDateUnchanged", bd["n_unchanged"], "board_response_dates.json: n_unchanged"),
        # exact one-sided 95% bound on the share of models with changed responses, valid only when none changed
        ("BoardDateUpperPct", math.ceil(1000 * (1 - 0.05 ** (1 / bd["n_checked"]))) / 10
         if bd["n_unchanged"] == bd["n_checked"] else None, "1 - 0.05^(1/n_checked), ceiled to 0.1%"),
        ("BoardDateConfPct", 95, "confidence level of BoardDateUpperPct (design constant)"),
        # the answer field against the reference solution's boxed answer (section 4, limitations)
        ("GoldConsistentN", gc["n_agree"], "gold_consistency.json: n_agree"),
        ("GoldFtItems", gf["n_affected"], "gold_ft.json: n_affected"),
        ("GoldFtLabels", gf["n_audited_labels_on_affected"], "gold_ft.json: n_audited_labels_on_affected"),
        ("GoldFtConcordant", gf["n_in_concordant_cells"], "gold_ft.json: n_in_concordant_cells"),
        ("GoldFtChanged", gf["n_verdicts_changed"], "gold_ft.json: n_verdicts_changed"),
        # where the disagreements fall, by model and subject (appendix I)
        ("TemplateMinPct", round(100 * ms["template_range_pct"][0], 1), "model_subject_table.json: summary"),
        ("TemplateMaxPct", round(100 * ms["template_range_pct"][1], 1), "model_subject_table.json: summary"),
        ("RankChangeModels", ms["models_rank_changes"], "model_subject_table.json: summary"),
        ("RankSpreadMedian", ms["rank_spread_median"], "model_subject_table.json: summary"),
        ("RankSpreadMax", ms["rank_spread_max"], "model_subject_table.json: summary"),
        ("DiscSubjectMinPct", round(100 * ms["discordant_subject_range"][0], 1), "model_subject_table.json: summary"),
        ("DiscSubjectMaxPct", round(100 * ms["discordant_subject_range"][1], 1), "model_subject_table.json: summary"),
        ("ExtrSubjectMinPct", round(100 * ms["extraction_share_subject_range"][0], 1),
         "model_subject_table.json: summary"),
        ("ExtrSubjectMaxPct", round(100 * ms["extraction_share_subject_range"][1], 1),
         "model_subject_table.json: summary"),
        # numerical checks (appendix B)
        ("TestsTotal", tests["total_cases"], "test_census.json: total_cases"),
        ("TestsFunctions", tests["total_functions"], "test_census.json: total_functions"),
        ("TestsCertification", tests["named"]["certification"], "test_census.json: named.certification"),
        ("TestsManuscriptClaims", tests["named"]["manuscript_claims"],
         "test_census.json: named.manuscript_claims"),
        ("TestsAnchor", tests["named"]["regression_anchor"], "test_census.json: named.regression_anchor"),
        # registered sensitivity analyses (appendices D and E)
        ("SensMvCombined", sens["mv_reference"]["combined"],
         "certify_results.json: K3.sensitivity.mv_reference.combined"),
        ("SensMvCorrectRejectAll", sens["mv_reference"]["audit_pooled"]["000"]["x_correct"],
         "certify_results.json: K3.sensitivity.mv_reference.audit_pooled.000"),
        ("SensMvCorrectAgreeAll", sens["mv_reference"]["audit_pooled"]["111"]["x_correct"],
         "certify_results.json: K3.sensitivity.mv_reference.audit_pooled.111"),
        ("SensDropCombined", sens["indeterminate_dropped"]["combined"],
         "certify_results.json: K3.sensitivity.indeterminate_dropped.combined"),
        ("SensDropNRejectAll", sens["indeterminate_dropped"]["audit_pooled"]["000"]["n"],
         "certify_results.json: K3.sensitivity.indeterminate_dropped.audit_pooled.000"),
        ("SensStringCombined", sens["string_semantics"]["combined"],
         "certify_results.json: K3.sensitivity.string_semantics.combined"),
        ("SensStringNAgreeAll", sens["string_semantics"]["audit_pooled"]["111"]["n"],
         "certify_results.json: K3.sensitivity.string_semantics.audit_pooled.111"),
        ("SensCpCombined", sens["clopper_pearson"]["combined"],
         "certify_results.json: K3.sensitivity.clopper_pearson.combined"),
        ("SensCpPrereg", sens["clopper_pearson"]["combined_eps0"][1],
         "certify_results.json: K3.sensitivity.clopper_pearson.combined_eps0"),
        ("SensSeventeenCombined", sens["models17"]["combined"],
         "certify_results.json: K3.sensitivity.models17.combined"),
        ("NumOrderingsSeventeen", len(m17) * (len(m17) - 1) // 2, "audit_k3_map.json: models17"),
        # existing-audit labels that fall in discordant cells (described, not used in certification)
        ("AuditDiscLabels", sum(disc_dom.values()), "audit_k3_map.json: items in non-concordant k3_sympy cells"),
        ("AuditDiscModels", len({m for m, _ in disc_dom}), "audit_k3_map.json: models with such items"),
        ("AuditDiscMedianPerModel", sorted(disc_dom.values())[len(disc_dom) // 2],
         "audit_k3_map.json: median labels per (model, discordant cell)"),
        # how the roster was chosen (roster_provenance.json, from the selection record)
        ("RosterCatalogued", rp["n_detail_datasets_catalogued"], "roster_provenance.json"),
        ("RosterSelected", rp["n_selected"], "roster_provenance.json"),
        ("RosterMaxPerOrg", rp["max_per_organisation"], "roster_provenance.json"),
        ("RosterOrgs", rp["n_organisations_selected"], "roster_provenance.json"),
        ("RosterScored", rp["n_scored_after_interrupted_sweep"], "roster_provenance.json"),
        ("RosterDeepestRank", rp["roster_deepest_rank"], "roster_provenance.json"),
        # where the roster sits on the final board (board_position.json)
        ("BoardModelsFinal", bpos["n_board_models_with_math"], "board_position.json"),
        ("RosterPctMedian", round(100 * bpos["roster_percentile_median"]), "board_position.json (percent)"),
        ("RosterBottomQuartile", bpos["roster_in_bottom_quartile"], "board_position.json"),
        ("RosterTopQuartile", bpos["roster_in_top_quartile"], "board_position.json"),
        # the verifier change at board scale, primary: the same run before and after the re-grade
        # (board_row_provenance.json, same_run_detail)
        ("BoardShiftModels", srd["board"]["n"], "board_row_provenance.json: same_run_detail.board.n"),
        ("BoardShiftTau", round(srd["board"]["kendall_tau_b"], 2), "board_row_provenance.json: same_run_detail"),
        ("BoardShiftDiscordantPct", pct(srd["board"]["discordant_pair_share"]),
         "board_row_provenance.json: same_run_detail"),
        ("RosterShiftModels", srd["roster"]["n"], "board_row_provenance.json: same_run_detail.roster.n"),
        ("RosterShiftTau", round(srd["roster"]["kendall_tau_b"], 2), "board_row_provenance.json: same_run_detail"),
        ("RosterShiftDiscordantPct", pct(srd["roster"]["discordant_pair_share"]),
         "board_row_provenance.json: same_run_detail"),
        # the reversal statistic spelled out: pairs, ties, and the denominator (Appendix G)
        ("BoardShiftPairs", srd["board"]["pair_counts"]["pairs"], "board_row_provenance.json: same_run_detail"),
        ("BoardShiftTiedOld", srd["board"]["pair_counts"]["tied_2024"], "board_row_provenance.json"),
        ("BoardShiftTiedNew", srd["board"]["pair_counts"]["tied_rescored"], "board_row_provenance.json"),
        ("BoardShiftPairsStrict", srd["board"]["pair_counts"]["strictly_ordered_by_both"],
         "board_row_provenance.json"),
        ("BoardShiftReversed", srd["board"]["pair_counts"]["reversed"], "board_row_provenance.json"),
        ("BoardShiftReversedAllPct", pct(srd["board"]["pair_counts"]["reversed_share_of_all_pairs"]),
         "board_row_provenance.json"),
        ("RosterShiftPairsStrict", srd["roster"]["pair_counts"]["strictly_ordered_by_both"],
         "board_row_provenance.json"),
        ("RosterShiftReversed", srd["roster"]["pair_counts"]["reversed"], "board_row_provenance.json"),
        # how each displayed score was traced, and the comparisons the same-run one replaces
        ("BoardBothModels", bshift["n_models_both"], "board_verifier_shift.json: n_models_both"),
        ("BoardSnapshotPct", pct(bshift["board"]["discordant_pair_share"]), "board_verifier_shift.json"),
        ("BoardAttributable", brp["classes"].get("attributable", 0), "board_row_provenance.json: classes"),
        ("BoardAttributablePct", pct(brp["reversal"]["attributable_only"]["reversed_share"]),
         "board_row_provenance.json: reversal.attributable_only"),
        ("BoardNoRegraded", brp["classes"].get("no re-graded file", 0), "board_row_provenance.json: classes"),
        ("BoardMismatch", brp["classes"].get("displayed score matches no re-graded file", 0),
         "board_row_provenance.json: classes"),
        ("BoardUnavailable", brp["classes"].get("unavailable", 0), "board_row_provenance.json: classes"),
        ("BoardSnapshotSameRun", brp["snapshot_score_equals_same_run_before"],
         "board_row_provenance.json: snapshot_score_equals_same_run_before"),
        ("BoardStaleRows", brp["displayed_rows_matching_no_regraded_file"], "board_row_provenance.json"),
        ("BoardRegradeDate", long_date(brp["regrade_day"]), "board_row_provenance.json: regrade_day"),
        # the historical moment each board count belongs to (board_timepoints.json)
        ("BoardPreRescoreDate", long_date(btime["results_before_rescoring"]["date"]), "board_timepoints.json"),
        ("BoardFinalTableDate", long_date(btime["final_public_table"]["date"]), "board_timepoints.json"),
        # template compliance and the taxonomy of disagreements (extraction_taxonomy.json)
        ("TemplateCompliant", tax["board"]["template_compliant"], "extraction_taxonomy.json"),
        ("TemplateCompliantPct", pct(tax["board"]["template_compliant"] / tax["board"]["n_responses"]),
         "extraction_taxonomy.json"),
        ("CreditedOthersNotEm", tax["board"]["credited_by_boxed_or_mv_not_2024"], "extraction_taxonomy.json"),
        ("CreditedOthersNotEmNoncompliant", tax["board"]["credited_by_boxed_or_mv_not_2024_noncompliant"],
         "extraction_taxonomy.json"),
        ("TaxExtraction", tax["taxonomy"]["top"].get("extraction", 0), "extraction_taxonomy.json"),
        ("TaxJudgement", tax["taxonomy"]["top"].get("judgement", 0), "extraction_taxonomy.json"),
        ("TaxRuntime", tax["taxonomy"]["top"].get("runtime", 0), "extraction_taxonomy.json"),
        ("TaxBoxOnly", tax["taxonomy"]["extraction_sub"].get("extraction:box", 0), "extraction_taxonomy.json"),
        ("TaxTemplateOnly", tax["taxonomy"]["extraction_sub"].get("extraction:template", 0),
         "extraction_taxonomy.json"),
        ("TaxTemplateBox", tax["taxonomy"]["extraction_sub"].get("extraction:template+box", 0),
         "extraction_taxonomy.json"),
        ("AuditReaderCorrect", ac["reader_correct|compliant|n"] + ac["reader_correct|noncompliant|n"],
         "extraction_taxonomy.json: audited_cross"),
        ("AuditReaderCorrectNoncompliant", ac["reader_correct|noncompliant|n"], "extraction_taxonomy.json"),
        ("AuditReaderCorrectNoncompliantEm", ac["reader_correct|noncompliant|em"], "extraction_taxonomy.json"),
        ("AuditReaderIncorrectNoncompliant", ac["reader_incorrect|noncompliant|n"], "extraction_taxonomy.json"),
        ("AuditReaderIncorrectNoncompliantSb", ac["reader_incorrect|noncompliant|sb_sympy"],
         "extraction_taxonomy.json"),
        ("AuditReaderIncorrectNoncompliantMv", ac["reader_incorrect|noncompliant|mv"], "extraction_taxonomy.json"),
        # meaning-preserving rewrites (perturbation.json)
        ("PertN", pert["control"]["n"], "perturbation.json"),
        ("PertSignoffEm", pert["signoff"]["em_rec"], "perturbation.json"),
        ("PertNoTemplateEm", pert["no_template"]["em_rec"], "perturbation.json"),
        ("PertNoBoxSb", pert["no_box"]["sb_sympy"], "perturbation.json"),
        ("PertNoBoxMv", pert["no_box"]["mv"], "perturbation.json"),
        # a fourth, learned verifier (xfinder_verdicts.json.gz, certify_results.json K4)
        ("XfCredited", sum(v["verdict"] for v in xfv["train"].values()), "xfinder_verdicts.json.gz: train"),
        ("XfAbstained", sum(not v["found"] for v in xfv["train"].values()), "xfinder_verdicts.json.gz: train"),
        ("NumDiscordantKFour", k4["train"]["n_discordant"], "certify_results.json: K4.train"),
        ("CertFixedKFourEpsZero", k4["train"]["fixed_board"]["certified_eps0"], "certify_results.json: K4.train"),
        ("CertFixedKFourPrereg", k4["train"]["fixed_board"]["certified_eps_preregistered"],
         "certify_results.json: K4.train"),
        ("CertFixedKFourPreregRelease", k4["release"]["fixed_board"]["certified_eps_preregistered"],
         "certify_results.json: K4.release"),
        # how the four-verifier analysis is built (Appendix, four verifiers)
        ("KFourCellsOccupied", sum(1 for v in k4["train"]["cells"].values() if v),
         "certify_results.json: K4.train.cells"),
        ("KFourMovedFromConcordant",
         k4["train"]["n_discordant"] - sum(len({r["em"], r["sb_sympy"], r["mv"]}) > 1 for r in rows),
         "certify_results.json: K4.train.n_discordant less the three-verifier discordant count"),
        ("KFourAuditNAgreeAll", k4["train"]["inputs"]["audit_pooled"]["1111"]["n"], "certify_results.json: K4.train"),
        ("KFourAuditNRejectAll", k4["train"]["inputs"]["audit_pooled"]["0000"]["n"],
         "certify_results.json: K4.train"),
        # external facts, each with its source recorded in the file
        ("BoardModelsRescored", ext["board_models_rescored"]["value"],
         "external_sources.json: board_models_rescored (maintainers' Math-Verify post)"),
        # audit design
        ("AuditBudget", op["budget"], "g3_operating_point.json: budget"),
        ("ScreenedModels", op["F"], "g3_operating_point.json: F"),
        ("ScreenNRejectAll", op["n0_by_cell"]["000"], "g3_operating_point.json: n0_by_cell"),
        ("ScreenNAgreeAll", op["n0_by_cell"]["111"], "g3_operating_point.json: n0_by_cell"),
        ("ScreenRho", op["rho"], "g3_operating_point.json: rho"),
        ("ScreenPowerPct", round(op["power"] * 100), "g3_operating_point.json: power"),
        ("ScreenPowerRejectAll", opt["falsification"]["000"]["power_rho20"], "g3_frontier.json: operating point"),
        ("ScreenPowerAgreeAll", opt["falsification"]["111"]["power_rho20"], "g3_frontier.json: operating point"),
        ("LabelsFalsification", op["labels_falsification"], "g3_operating_point.json"),
        ("LabelsCertification", op["labels_certification"], "g3_operating_point.json"),
        ("ProjectedCertPrereg", opt["certified_uniform_eps0.01"]["median"],
         "g3_frontier.json: operating point (PROJECTED)"),
        # how widely the same design choice is made (harness census)
        ("LmEvalGenerative", prev["lm-evaluation-harness"][1], "harness_census/prevalence.json"),
        ("LmEvalCoerce", prev["lm-evaluation-harness"][2], "harness_census/prevalence.json"),
        ("LmEvalCoercePct", prev["lm-evaluation-harness"][3], "harness_census/prevalence.json"),
        ("OpenCompassGenerative", prev["OpenCompass"][1], "harness_census/prevalence.json"),
        ("OpenCompassCoerce", prev["OpenCompass"][2], "harness_census/prevalence.json"),
        ("OpenCompassCoercePct", prev["OpenCompass"][3], "harness_census/prevalence.json"),
        ("LightevalGenerative", light["n_generative"], "harness_census/census_lighteval.json"),
        ("LightevalCoerce", light["n_vulnerable"], "harness_census/census_lighteval.json"),
        ("SimpleEvalsStringMatch", simple["n_string_match_modules"], "harness_census/simple_evals_census.json"),
        ("SimpleEvalsCoerce", simple["n_vulnerable"], "harness_census/simple_evals_census.json"),
        ("CensusValidTestable", valid["vulnerable"]["n_testable"], "harness_census/census_validation.json"),
        ("CensusValidConfirmed", valid["vulnerable"]["n_agreeing_with_census"], "harness_census/census_validation.json"),
        ("CensusControlN", valid["control"]["n_sampled"], "harness_census/census_validation.json"),
        ("CensusControlClean", valid["control"]["not_vulnerable"], "harness_census/census_validation.json"),
    ]
    M += labeller_macros(lab)
    M += projection_macros()
    M += post_label_macros(K3, eps0)  # empty until the registered audit or the census is labelled
    body = ("% Generated by scripts/fb_macros_aistats.py -- do not edit by hand.\n"
            "% Every value is read from a committed result file; see CLAIM_LEDGER_AISTATS.md.\n"
            + "".join(f"\\newcommand{{\\{name}}}{{{fmt(value)}}}\n" for name, value, _ in M))
    TEX.parent.mkdir(parents=True, exist_ok=True)
    TEX.write_text(body, encoding="utf-8")  # one copy: the build and the guard both read this file
    with open(LEDGER, "w", encoding="utf-8") as f:
        f.write("# Claim ledger (AISTATS submission)\n\n")
        f.write("Every number the manuscript states is a macro from `manuscript/tmlr/macros.tex`,\n"
                "generated by `scripts/fb_macros_aistats.py`. Projections are marked.\n\n")
        f.write("| macro | value | source |\n|---|---|---|\n")
        for name, value, src in M:
            tag = " **(PROJECTED)**" if "PROJECTED" in src else ""
            f.write(f"| `\\{name}` | {fmt(value)} | `{src}`{tag} |\n")
    print(f"wrote {TEX} ({len(M)} macros) and {LEDGER}")


if __name__ == "__main__":
    main()
