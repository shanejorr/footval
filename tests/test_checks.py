import copy

from conftest import CELLS, normal_strategy, skew_strategy, strategy, t_strategy, valid_response
from scipy import stats as st

from footbench.checks import CHECK_NAMES, check_instance
from footbench.distributions import skewnorm_mode

TOL = 0.10


def check(parsed, mode="direct"):
    return check_instance(parsed, mode, TOL)


def mixed_response():
    """4 normals, one student_t, one skew_normal — all scipy-derived, all valid."""
    strategies = [
        normal_strategy(b, t, f"Strategy {i}", mpv=0.2 * i, half_width=1.0 + 0.1 * i)
        for i, (b, t) in enumerate(CELLS[:4])
    ]
    strategies.append(t_strategy(*CELLS[4], "Heavy tail pick", mpv=0.4, half_width=1.5, nu=4))
    strategies.append(skew_strategy(*CELLS[5], "Skewed pick", xi=0.3, omega=1.2, alpha=4.0))
    return valid_response(strategies)


# --- whole-instance behavior ---------------------------------------------------


def test_valid_response_passes_all_checks():
    report = check(mixed_response())
    for name in CHECK_NAMES:
        assert report[name] is True, name


def test_unparseable_fails_everything():
    report = check(None, "failed")
    for name in CHECK_NAMES:
        assert report[name] is False, name
    assert report["detail"]["parse_mode"] == "failed"


def test_fenced_parse_mode_recorded_but_valid():
    report = check(valid_response(), "fenced")
    assert report["json_valid"] is True
    assert report["detail"]["parse_mode"] == "fenced"


def test_missing_top_level_key_fails_json_valid():
    resp = valid_response()
    del resp["plot_script"]
    report = check(resp)
    assert report["json_valid"] is False
    assert "plot_script" in report["detail"]["json"]["missing_keys"]
    for name in CHECK_NAMES:
        assert report[name] is False


def test_top_level_not_object_fails():
    assert check([1, 2, 3])["json_valid"] is False


def test_strategies_not_a_list_fails():
    resp = valid_response()
    resp["strategies"] = {"oops": 1}
    assert check(resp)["json_valid"] is False


# --- grid ----------------------------------------------------------------------


def test_five_strategies_fail_grid():
    resp = valid_response()
    resp["strategies"] = resp["strategies"][:5]
    report = check(resp)
    assert report["grid_complete"] is False
    assert len(report["detail"]["grid"]["missing_cells"]) == 1
    # the other checks are independent of grid completeness
    assert report["intervals_ok"] is True
    assert report["params_reproduce_ok"] is True


def test_seven_strategies_fail_grid():
    resp = valid_response()
    resp["strategies"].append(normal_strategy("analytics", "offensive", "Extra pick"))
    report = check(resp)
    assert report["grid_complete"] is False
    assert "analytics/offensive" in report["detail"]["grid"]["duplicate_cells"]


def test_duplicate_and_missing_cell():
    resp = valid_response()
    resp["strategies"][5]["bucket"] = resp["strategies"][0]["bucket"]
    resp["strategies"][5]["type"] = resp["strategies"][0]["type"]
    report = check(resp)
    grid = report["detail"]["grid"]
    assert report["grid_complete"] is False
    assert grid["duplicate_cells"] and grid["missing_cells"]


def test_duplicate_titles_case_and_whitespace_insensitive():
    resp = valid_response()
    resp["strategies"][1]["title"] = "  STRATEGY 0 "
    report = check(resp)
    assert report["grid_complete"] is False
    assert report["detail"]["grid"]["duplicate_titles"] == ["strategy 0"]


def test_invalid_bucket_literal_fails_grid():
    resp = valid_response()
    resp["strategies"][0]["bucket"] = "Analytics"  # exact lowercase required
    report = check(resp)
    assert report["grid_complete"] is False
    assert any("invalid cell" in i for i in report["detail"]["grid"]["issues"])


# --- interval sanity -------------------------------------------------------------


def _interval_detail(report, idx=0):
    return report["detail"]["strategies"][idx]


def test_low_above_high_fails():
    resp = valid_response()
    bs = resp["strategies"][0]["prior"]["belief_summaries"]
    bs["interval_95_low"], bs["interval_95_high"] = 1.0, -1.0
    report = check(resp)
    assert report["intervals_ok"] is False
    assert _interval_detail(report)["interval_sane"] is False


def test_low_equal_high_fails():
    resp = valid_response()
    bs = resp["strategies"][0]["prior"]["belief_summaries"]
    bs["interval_95_low"] = bs["interval_95_high"]
    assert check(resp)["intervals_ok"] is False


def test_mpv_outside_interval_fails():
    resp = valid_response()
    resp["strategies"][0]["prior"]["belief_summaries"]["most_plausible_value"] = 99.0
    assert check(resp)["intervals_ok"] is False


def test_mpv_on_boundary_passes_interval_check():
    s = strategy(
        "analytics",
        "game_management",
        "Boundary",
        "normal",
        {"mu": -1.0, "sigma": 2 / 3.92},
        mpv=-1.0,
        low=-1.0,
        high=1.0,
    )
    resp = valid_response()
    resp["strategies"][0] = s
    report = check(resp)
    assert _interval_detail(report)["interval_sane"] is True


def test_string_typed_interval_fails():
    resp = valid_response()
    resp["strategies"][0]["prior"]["belief_summaries"]["interval_95_low"] = "-1.0"
    assert check(resp)["intervals_ok"] is False


def test_bool_typed_value_rejected():
    resp = valid_response()
    resp["strategies"][0]["prior"]["belief_summaries"]["most_plausible_value"] = True
    assert check(resp)["intervals_ok"] is False


# --- family ---------------------------------------------------------------------


def test_capitalized_family_fails():
    resp = valid_response()
    resp["strategies"][0]["prior"]["distribution_family"]["name"] = "Normal"
    report = check(resp)
    assert report["family_ok"] is False
    assert report["params_reproduce_ok"] is False  # not evaluable without a valid family


def test_unsupported_family_fails():
    resp = valid_response()
    resp["strategies"][0]["prior"]["distribution_family"]["name"] = "lognormal"
    assert check(resp)["family_ok"] is False


def test_missing_family_name_fails():
    resp = valid_response()
    del resp["strategies"][0]["prior"]["distribution_family"]["name"]
    assert check(resp)["family_ok"] is False


# --- parameter reproduction: normal ----------------------------------------------


def test_normal_exact_derivation_passes():
    assert check(valid_response())["params_reproduce_ok"] is True


def test_normal_sigma_inflated_beyond_tolerance_fails():
    resp = valid_response()
    params = resp["strategies"][0]["prior"]["parameters"]
    params["sigma"] *= 1.21  # endpoint error = 0.105 x width > tol
    report = check(resp)
    assert report["params_reproduce_ok"] is False
    detail = _interval_detail(report)
    assert detail["params_ok"] is False
    assert detail["errors"]["low"] > detail["tol_abs"]


def test_normal_error_just_below_tolerance_passes():
    # reported [-1, 1] (width 2, tol 0.2); recomputed endpoints at +/-1.2 - eps
    sigma = (1.2 - 1e-9) / 1.96
    s = strategy(
        "analytics",
        "game_management",
        "Near boundary",
        "normal",
        {"mu": 0.0, "sigma": sigma},
        mpv=0.0,
        low=-1.0,
        high=1.0,
    )
    resp = valid_response()
    resp["strategies"][0] = s
    assert check(resp)["params_reproduce_ok"] is True


def test_normal_error_just_above_tolerance_fails():
    sigma = 1.201 / 1.96
    s = strategy(
        "analytics",
        "game_management",
        "Past boundary",
        "normal",
        {"mu": 0.0, "sigma": sigma},
        mpv=0.0,
        low=-1.0,
        high=1.0,
    )
    resp = valid_response()
    resp["strategies"][0] = s
    assert check(resp)["params_reproduce_ok"] is False


def test_sigma_zero_or_negative_fails():
    for bad in (0.0, -0.5):
        resp = valid_response()
        resp["strategies"][0]["prior"]["parameters"]["sigma"] = bad
        report = check(resp)
        assert report["params_reproduce_ok"] is False
        assert "sigma" in _interval_detail(report)["reason"]


def test_near_zero_center_passes_width_relative_tolerance():
    # honest weak cell: center 0, interval [-1, 1] — must NOT be penalized
    s = strategy(
        "analytics",
        "game_management",
        "Weak cell",
        "normal",
        {"mu": 0.0, "sigma": 2 / 3.92},
        mpv=0.0,
        low=-1.0,
        high=1.0,
    )
    resp = valid_response()
    resp["strategies"][0] = s
    assert check(resp)["params_reproduce_ok"] is True


def test_center_shift_fails():
    resp = valid_response()
    s = resp["strategies"][0]
    width = (
        s["prior"]["belief_summaries"]["interval_95_high"]
        - s["prior"]["belief_summaries"]["interval_95_low"]
    )
    s["prior"]["parameters"]["mu"] += 0.11 * width
    assert check(resp)["params_reproduce_ok"] is False


# --- parameter reproduction: student_t --------------------------------------------


def test_student_t_correct_critical_values_pass():
    for nu in (4, 30):
        resp = valid_response()
        resp["strategies"][0] = t_strategy(
            "analytics", "game_management", "Strategy 0", mpv=0.5, half_width=1.0, nu=nu
        )
        assert check(resp)["params_reproduce_ok"] is True, nu


def test_student_t_derived_with_normal_critical_value_fails():
    # sigma from (high-low)/(2*1.96) instead of the t critical value at nu=4
    low, high, mpv, nu = -0.5, 1.5, 0.5, 4
    s = strategy(
        "analytics",
        "game_management",
        "Wrong crit",
        "student_t",
        {"nu": nu, "mu": mpv, "sigma": (high - low) / (2 * 1.96)},
        mpv=mpv,
        low=low,
        high=high,
    )
    resp = valid_response()
    resp["strategies"][0] = s
    assert check(resp)["params_reproduce_ok"] is False


def test_student_t_missing_nu_fails_with_key_detail():
    resp = valid_response()
    resp["strategies"][0] = t_strategy("analytics", "game_management", "Strategy 0")
    del resp["strategies"][0]["prior"]["parameters"]["nu"]
    report = check(resp)
    assert report["params_reproduce_ok"] is False
    assert "nu" in _interval_detail(report)["reason"]


def test_student_t_negative_nu_fails():
    resp = valid_response()
    resp["strategies"][0] = t_strategy("analytics", "game_management", "Strategy 0")
    resp["strategies"][0]["prior"]["parameters"]["nu"] = -1
    assert check(resp)["params_reproduce_ok"] is False


# --- parameter reproduction: skew_normal -------------------------------------------


def test_skew_normal_scipy_derived_passes():
    resp = valid_response()
    resp["strategies"][0] = skew_strategy(
        "analytics", "game_management", "Strategy 0", xi=0.3, omega=1.2, alpha=4.0
    )
    assert check(resp)["params_reproduce_ok"] is True


def test_skew_normal_shifted_endpoint_fails():
    resp = valid_response()
    s = skew_strategy("analytics", "game_management", "Strategy 0", xi=0.3, omega=1.2, alpha=4.0)
    bs = s["prior"]["belief_summaries"]
    bs["interval_95_high"] += 0.2 * (bs["interval_95_high"] - bs["interval_95_low"])
    resp["strategies"][0] = s
    assert check(resp)["params_reproduce_ok"] is False


def test_skew_normal_xi_reported_as_mpv_fails_center():
    xi, omega, alpha = 0.3, 1.2, 4.0
    dist = st.skewnorm(alpha, loc=xi, scale=omega)
    low, high = float(dist.ppf(0.025)), float(dist.ppf(0.975))
    mode = skewnorm_mode(xi, omega, alpha)
    assert abs(mode - xi) > TOL * (high - low)  # precondition: xi is a bad mpv here
    s = strategy(
        "analytics",
        "game_management",
        "Xi as mpv",
        "skew_normal",
        {"xi": xi, "omega": omega, "alpha": alpha},
        mpv=xi,
        low=low,
        high=high,
    )
    resp = valid_response()
    resp["strategies"][0] = s
    report = check(resp)
    assert report["params_reproduce_ok"] is False
    assert _interval_detail(report)["errors"]["center"] > _interval_detail(report)["tol_abs"]


def test_skewnorm_mode_alpha_zero_is_xi():
    assert abs(skewnorm_mode(0.5, 1.0, 0.0) - 0.5) < 1e-3


def test_skew_normal_extreme_alpha_converges():
    resp = valid_response()
    resp["strategies"][0] = skew_strategy(
        "analytics", "game_management", "Strategy 0", xi=0.0, omega=1.0, alpha=50.0
    )
    assert check(resp)["params_reproduce_ok"] is True


# --- parameter keys ---------------------------------------------------------------


def test_wrong_key_set_for_family_fails():
    resp = valid_response()
    s = resp["strategies"][0]
    s["prior"]["distribution_family"]["name"] = "student_t"  # but params are mu/sigma
    report = check(resp)
    assert report["params_reproduce_ok"] is False
    assert "nu" in _interval_detail(report)["reason"]


def test_extra_param_keys_ignored():
    resp = valid_response()
    resp["strategies"][0]["prior"]["parameters"]["nu"] = 7  # stray key on a normal
    assert check(resp)["params_reproduce_ok"] is True


# --- rollup ------------------------------------------------------------------------


def test_one_failing_strategy_fails_instance_with_detail():
    resp = valid_response()
    resp["strategies"][3] = copy.deepcopy(resp["strategies"][3])
    resp["strategies"][3]["prior"]["parameters"]["sigma"] *= 1.5
    report = check(resp)
    assert report["params_reproduce_ok"] is False
    flags = [d["params_ok"] for d in report["detail"]["strategies"]]
    assert flags.count(False) == 1 and flags[3] is False
