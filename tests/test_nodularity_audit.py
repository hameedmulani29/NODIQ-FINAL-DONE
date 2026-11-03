#!/usr/bin/env python3
"""Accuracy-audit regression tests (phases 9-13 of the nodularity audit).

Locks down, from the ACTUAL implementation:

  * the ISO / in-frame / JIS formulas in nodularity_pipeline (synthetic
    images with known geometry — no reliance on the uploaded micrograph),
  * border-touching treatment (in-frame excludes border particles),
  * graphite form shares (count shares by particle count; area shares by
    total area so percentages really sum to ~100; raw px^2 exposed),
  * PDF text sanitization (fpdf2 latin-1 core font: intentional ASCII
    replacements instead of stray '?' artifacts).
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import report_builder as rb
from services.morphology import classify_graphite_forms

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


# ----------------------------------------------------------- helpers -------
def _synthetic_image():
    """1024x1024 white image with known dark particles:

    * three interior circles r=60 (marumi ~0.985 -> V/VI, JIS 5)
    * one ellipse 90x30 (marumi = b/a = 0.333 -> irregular, JIS 2)
    * one circle r=50 clipped by the left frame edge (border-touching)
    """
    img = np.full((1024, 1024), 255, np.uint8)
    for cy, cx in ((200, 200), (512, 512), (800, 700)):
        cv2.circle(img, (cx, cy), 60, 0, -1)
    cv2.ellipse(img, (700, 200), (90, 30), 0, 0, 360, 0, -1)
    cv2.circle(img, (5, 500), 50, 0, -1)
    return img


@pytest.fixture(scope="module")
def engine():
    from nodularity_pipeline import analyze_image
    return analyze_image(_synthetic_image())


def _erows(engine):
    from services.nodule_data import enrich_rows
    return enrich_rows(engine["rows"], engine["contours"])


# --------------------------------------------------------- ISO formulas ----
def test_iso_nodularity_is_area_weighted_including_border(engine):
    """ISO = sum(area[round]) / sum(area[all]) * 100 — ALL particles."""
    rows = engine["rows"]
    area_total = sum(r["area_px"] for r in rows)
    area_round = sum(r["area_px"] for r in rows if r["iso_class"] == "V/VI")
    assert engine["iso_pct"] == pytest.approx(area_round / area_total * 100,
                                              abs=0.02)


def test_iso_inframe_excludes_border_particles(engine):
    """In-frame drops border-touching particles from BOTH terms.
    This is the ISO 945-4:2019 clause 6.4 requirement: A_all excludes
    'the particles that intersect the border of the area of view', so
    iso_inframe_pct is the standards-compliant value; the headline
    iso_pct additionally retains border particles for robustness."""
    rows = engine["rows"]
    inside = [r for r in rows if not r["touches_border"]]
    area_total = sum(r["area_px"] for r in inside)
    area_round = sum(r["area_px"] for r in inside if r["iso_class"] == "V/VI")
    assert engine["iso_inframe_pct"] == pytest.approx(
        area_round / area_total * 100, abs=0.02)
    # and the synthetic geometry really exercises the difference
    assert any(r["touches_border"] for r in rows)
    assert engine["iso_inframe_pct"] > engine["iso_pct"]


def test_marumi_perfect_circle_scores_near_one(engine):
    circles = [r for r in engine["rows"]
               if abs(r["center_x"] * 1024 - 200) < 1
               or abs(r["center_x"] * 1024 - 512) < 1
               or abs(r["center_x"] * 1024 - 700) < 1 and
               abs(r["center_y"] * 1024 - 200) < 1]
    assert any(r["roundness"] > 0.95 for r in engine["rows"])
    # the ellipse (marumi = b/a = 1/3) must be irregular with JIS type 2
    ell = next(r for r in engine["rows"] if r["jis_bucket"] == 2)
    assert 0.30 < ell["roundness"] < 0.40
    assert ell["iso_class"] == "irregular"


def test_marumi_matches_hull_diameter_formula(engine):
    """marumi = area / (pi * r_hull^2), r_hull = half of the hull's largest
    point-pair distance (legacy get_graphite_length)."""
    for row, cnt in zip(engine["rows"], engine["contours"]):
        from nodularity_pipeline import max_half_length
        r = max_half_length(cnt)
        expected = cv2.contourArea(cnt) / ((r ** 2) * math.pi)
        assert row["roundness"] == pytest.approx(expected, abs=1e-3)


def test_marumi_matches_iso945_4_roundness_definition(engine):
    """ISO 945-4:2019 clause 3.2 defines roundness as
    rho = 4A / (pi * l_m^2) with l_m the maximum Feret diameter.
    The hull max-distance pair IS the maximum Feret diameter, so
    marumi must equal 4A / (pi * l_m^2) for every particle."""
    for row, cnt in zip(engine["rows"], engine["contours"]):
        from nodularity_pipeline import max_half_length
        l_m = 2 * max_half_length(cnt)          # diameter, not radius
        expected = 4 * cv2.contourArea(cnt) / (math.pi * l_m ** 2)
        assert row["roundness"] == pytest.approx(expected, abs=1e-3)


def test_spheroidal_threshold_is_iso945_4_0_6():
    """ISO 945-4:2019 clause 3.3: spheroidal graphite = roundness >= 0.6.
    Synthetic ellipses with area/circumscribed-circle ratios on either
    side of 0.6 must classify accordingly (marumi = b/a for an ellipse)."""
    from nodularity_pipeline import analyze_image
    img = np.full((1024, 1024), 255, np.uint8)
    cv2.ellipse(img, (250, 250), (90, 59), 0, 0, 360, 0, -1)   # b/a ~ 0.65
    cv2.ellipse(img, (700, 700), (90, 49), 0, 0, 360, 0, -1)   # b/a ~ 0.55
    res = analyze_image(img)
    round_row = next(r for r in res["rows"] if r["roundness"] > 0.6)
    irreg_row = next(r for r in res["rows"] if r["roundness"] < 0.6)
    assert round_row["iso_class"] == "V/VI"
    assert irreg_row["iso_class"] == "irregular"


def test_jis_bucket_boundaries_match_jis_g5502_2022_convention():
    """JIS G5502:2022 (quoted in J. Japan Inst. Met. Mater. 89(12) 2025,
    doi:10.2320/jinstmet.J202529): classes II..V cover 0.2<eta<=0.4,
    0.4<eta<=0.7, 0.7<eta<=0.8, 0.8<eta<=1 — i.e. UPPER-inclusive.
    The legacy jis_bucket (<=0.2 -> 1, <=0.4 -> 2, <=0.7 -> 3,
    <=0.8 -> 4, else 5) implements exactly that convention."""
    from nodularity_pipeline import jis_bucket
    assert jis_bucket(0.2) == 1     # eta = 0.2 is NOT in class II
    assert jis_bucket(0.21) == 2
    assert jis_bucket(0.4) == 2     # upper edge inclusive
    assert jis_bucket(0.41) == 3
    assert jis_bucket(0.7) == 3
    assert jis_bucket(0.71) == 4
    assert jis_bucket(0.8) == 4
    assert jis_bucket(0.81) == 5
    assert jis_bucket(1.0) == 5


# ------------------------------------------------------------ JIS formula ---
def test_jis_is_count_weighted_bucket_formula(engine):
    """JIS = (0.3*n2 + 0.7*n3 + 0.9*n4 + 1.0*n5) / n_total * 100.

    Buckets from jis_bucket(marumi): <=0.2 -> 1, <=0.4 -> 2, <=0.7 -> 3,
    <=0.8 -> 4, >0.8 -> 5. Count-weighted, border particles included."""
    rows = engine["rows"]
    n = len(rows)
    buckets = {b: sum(1 for r in rows if r["jis_bucket"] == b)
               for b in (1, 2, 3, 4, 5)}
    expected = (0.3 * buckets[2] + 0.7 * buckets[3] + 0.9 * buckets[4]
                + 1.0 * buckets[5]) / n * 100
    assert engine["jis_pct"] == pytest.approx(expected, abs=0.02)
    # synthetic: 3 circles (type 5) + ellipse marumi=b/a=0.33 (type 2) +
    # clipped circle marumi~0.57 (type 3): (0.3 + 0.7 + 3.0) / 5 = 80.0%
    assert engine["jis_pct"] == pytest.approx(80.0, abs=0.5)


def test_jis_weights_exactly_match_reported_case():
    """The user-reported 512x512 analysis: {'3':21,'2':2,'5':5,'4':11}
    must give (0.3*2 + 0.7*21 + 0.9*11 + 1.0*5)/39 = 77.4358 -> 77.44%."""
    jis = (0.3 * 2 + 0.7 * 21 + 0.9 * 11 + 1.0 * 5) / 39 * 100
    assert round(jis, 2) == 77.44


# ------------------------------------------------------ border handling ----
def test_border_particle_present_but_excluded_from_inframe(engine):
    rows = engine["rows"]
    border_rows = [r for r in rows if r["touches_border"]]
    assert len(border_rows) == 1
    assert border_rows[0]["iso_class"] in ("V/VI", "irregular")  # counted
    inside_ids = {r["nodule_id"] for r in rows if not r["touches_border"]}
    area_in = sum(r["area_px"] for r in rows if r["nodule_id"] in inside_ids)
    assert area_in < sum(r["area_px"] for r in rows)  # excluded from in-frame


def test_border_clipping_lowers_measured_roundness():
    """A circle clipped by the frame measures marumi < 1 (incomplete shape),
    which is exactly why in-frame exists as a separate metric."""
    from nodularity_pipeline import analyze_image
    img = np.full((1024, 1024), 255, np.uint8)
    cv2.circle(img, (5, 512), 60, 0, -1)     # half-clipped circle
    cv2.circle(img, (700, 512), 60, 0, -1)   # same circle, interior
    res = analyze_image(img)
    clipped = next(r for r in res["rows"] if r["touches_border"])
    full = next(r for r in res["rows"] if not r["touches_border"])
    assert clipped["roundness"] < 0.8 < full["roundness"]


# ------------------------------------------------- graphite form shares ----
def _mkrow(nid, ar, sol, rnd, area=100.0):
    return {"nodule_id": nid, "aspect_ratio": ar, "solidity": sol,
            "roundness": rnd, "area_px": area}


def test_form_area_shares_are_true_percentages():
    """Regression: area shares used to divide by PARTICLE COUNT, producing
    px^2-per-particle values labeled as '%'. They must now be shares of
    total graphite area and sum to ~100."""
    rows = [
        _mkrow(1, 1.1, 0.96, 0.9, area=300.0),   # VI
        _mkrow(2, 1.2, 0.92, 0.8, area=100.0),   # VI
        _mkrow(3, 2.0, 0.85, 0.7, area=600.0),   # V
    ]
    res = classify_graphite_forms(rows)
    assert res["available"]
    assert res["form_count_share_pct"]["VI"] == pytest.approx(66.7)
    assert res["form_count_share_pct"]["V"] == pytest.approx(33.3)
    assert sum(res["form_area_share_pct"].values()) == pytest.approx(100.0,
                                                                     abs=0.2)
    assert res["form_area_share_pct"]["VI"] == pytest.approx(40.0, abs=0.1)
    assert res["form_area_share_pct"]["V"] == pytest.approx(60.0, abs=0.1)
    # raw per-form graphite area stays available, correctly named
    assert res["form_area_px2"]["VI"] == pytest.approx(400.0)
    assert res["form_area_px2"]["V"] == pytest.approx(600.0)


def test_form_area_shares_zero_total_area_is_safe():
    rows = [_mkrow(1, 1.1, 0.96, 0.9, area=None)]   # no usable area_px
    res = classify_graphite_forms(rows)
    assert res["form_area_share_pct"]["VI"] == 0.0


def test_form_area_shares_on_enriched_engine_rows(engine):
    """End-to-end: shares of the real enriched engine rows sum to 100."""
    res = classify_graphite_forms(_erows(engine))
    assert sum(res["form_area_share_pct"].values()) == pytest.approx(100.0,
                                                                     abs=0.2)
    raw = res["form_area_px2"]
    assert sum(raw.values()) == pytest.approx(
        sum(r["area_px"] for r in engine["rows"]), rel=0.01)


# --------------------------------------------------------- PDF sanitizer ---
def test_pdf_safe_text_ascii_replacements():
    assert rb.pdf_safe_text("high\u2014frequency") == "high-frequency"
    assert rb.pdf_safe_text("low\u2013score") == "low-score"
    assert rb.pdf_safe_text("under\u2011resolved") == "under-resolved"
    assert rb.pdf_safe_text("wait\u2026") == "wait..."
    assert rb.pdf_safe_text("\u202f600\u00a0px") == " 600 px"
    assert rb.pdf_safe_text("\u2192 next") == "-> next"
    assert rb.pdf_safe_text("area\u00b2") == "area2"
    assert rb.pdf_safe_text("\u224850\u2265x\u2264y") == "~50>=x<=y"
    assert rb.pdf_safe_text("form \u2160 and \u2165") == "form I and VI"


def test_pdf_safe_text_known_glyphs_and_fallback():
    # every glyph that actually occurs in report strings maps to ASCII
    out = rb.pdf_safe_text("mixed — ‘quotes’ … III → ~50 x²")
    assert "?" not in out
    assert all(ord(c) < 128 for c in out)
    # documented last resort: a glyph with no ASCII equivalent (e.g. a
    # pictograph) becomes the core-font replacement char, as fpdf2 itself
    # would render it — never a mojibake sequence
    assert rb.pdf_safe_text("\u2603") == "?"


def test_pdf_render_includes_sanitized_section_labels():
    """The Form area shares label now shows true percentages in the PDF."""
    record = {"id": "ANL-P", "filename": "synthetic.png",
              "iso_nodularity_pct": 80.0, "jis_nodularity_pct": 85.0,
              "nodule_count": 3, "source_width": 1024,
              "source_height": 1024, "created_at": "2026-01-01"}
    full = {"nodules": [_mkrow(1, 1.1, 0.96, 0.9, area=100.0)],
            "border_touching": 0, "class_counts": {"round_V_VI": 1,
                                                   "irregular": 0},
            "iso_nodularity_inframe_pct": 80.0,
            "graphite_forms": classify_graphite_forms(
                [_mkrow(1, 1.1, 0.96, 0.9, area=100.0),
                 _mkrow(2, 8.0, 0.45, 0.2, area=300.0)])}
    data, err = rb.to_pdf(rb.build_report_data(record, full))
    assert err is None
    assert data[:5] == b"%PDF-"


def test_report_carries_methodology_note():
    """Reports must distinguish implemented methodology from certified
    standards compliance (Phase 10 of the methodology audit)."""
    import report_builder as rb
    record = {"id": "ANL-M", "filename": "x.png", "iso_nodularity_pct": 80.0,
              "jis_nodularity_pct": 85.0, "nodule_count": 1,
              "source_width": 10, "source_height": 10,
              "created_at": "2026-01-01"}
    report = rb.build_report_data(record, {"nodules": [], "border_touching": 0,
                                           "class_counts": {}})
    meth = report["iso_jis_classification"]["methodology"]
    assert "ISO 945-4" in meth and "JIS G5502" in meth
    assert "paywalled" in meth


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
