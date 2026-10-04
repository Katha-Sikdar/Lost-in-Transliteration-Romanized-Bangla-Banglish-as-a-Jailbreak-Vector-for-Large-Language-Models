"""Offline tests: run with `python -m pytest`. No API calls are made."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from banglishjail import VERSIONS
from banglishjail import cmi, data, judge, labels, noise, run_eval, stats
from banglishjail.clients import DummyClient, extract_json
from banglishjail.defenses import sft_data
from banglishjail.guard_eval import DummyGuard, evaluate, summarize
from banglishjail.io import latest_by_key, read_jsonl
from banglishjail.mech.refusal_direction import project_seed, select_layer

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "data" / "templates" / "seeds_template.csv"
DUMMY_CONFIG = ROOT / "configs" / "dummy.yaml"


def make_seeds(tmp_path, n_harmful=6, n_benign=3):
    rows = []
    for i in range(n_harmful + n_benign):
        benign = i >= n_harmful
        row = {"id": f"S{i:03d}", "category": "cat_a" if i % 2 else "cat_b",
               "is_benign_control": str(benign).lower(), "split": ""}
        for v in VERSIONS:
            row[v] = f"{v} text number {i} kemon bhalo tumi"
        rows.append(row)
    path = tmp_path / "seeds.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# --- data -------------------------------------------------------------------

def test_template_is_valid():
    df = data.load_seeds(TEMPLATE)
    assert data.validate(df) == []


def test_validate_reports_problems(tmp_path):
    df = data.load_seeds(make_seeds(tmp_path))
    df.loc[1, "id"] = df.loc[0, "id"]
    df.loc[2, "bn"] = ""
    problems = data.validate(df)
    assert any("duplicate" in p for p in problems)
    assert any("empty version bn" in p for p in problems)


def test_assign_splits_is_stable_and_keeps_existing(tmp_path):
    df = data.load_seeds(make_seeds(tmp_path, n_harmful=10, n_benign=10))
    df.loc[0, "split"] = "test"
    a = data.assign_splits(df, 0.3)
    b = data.assign_splits(df, 0.3)
    assert list(a["split"]) == list(b["split"])
    assert a.loc[0, "split"] == "test"
    assert set(a["split"]) == {"train", "test"}


def test_expand_is_long_format(tmp_path):
    df = data.assign_splits(data.load_seeds(make_seeds(tmp_path)))
    records = data.expand(df)
    assert len(records) == len(df) * len(VERSIONS)
    assert {"seed_id", "version", "prompt", "en_reference"} <= set(records[0])


# --- noise and CMI -----------------------------------------------------------

def test_noise_rules():
    assert noise.perturb_word("kemon") == ("kmn", "abbreviation")
    assert noise.perturb_word("bhai")[0] == "vai"
    assert noise.perturb_word("korechi") == ("krch" + "i", "vowel_drop")
    assert noise.perturb_word("Tumi")[0] == "Tmi"


def test_perturb_levels_and_determinism():
    text = "tumi kemon acho, bhalo?"
    assert noise.perturb(text, 0.0)[0] == text
    full, rules = noise.perturb(text, 1.0)
    assert full == "tmi kmn acho, vlo?"
    assert rules
    assert noise.perturb(text, 0.5, seed=3) == noise.perturb(text, 0.5, seed=3)
    with pytest.raises(ValueError):
        noise.perturb(text, 1.5)


def test_noise_level():
    assert noise.levenshtein("kemon", "kmn") == 2
    assert noise.noise_level("kemon", "kemon") == 0
    assert 0 < noise.noise_level("tumi kemon", "tmi kmn") < 1


def test_cmi():
    assert cmi.cmi(cmi.parse_tagged("ami/bn bhalo/bn achi/bn")) == 0
    mixed = cmi.parse_tagged("bro/en tumi/bn how/en acho/bn ?/univ")
    assert cmi.cmi(mixed) == pytest.approx(50.0)
    with pytest.raises(ValueError):
        cmi.parse_tagged("untagged")
    tags = cmi.heuristic_tags("bro tumi how acho?", english_words={"bro", "how", "to"})
    assert [t for _, t in tags] == ["en", "bn", "en", "bn", "univ"]


# --- run, judge, labels, stats (dummy end-to-end) ----------------------------

@pytest.fixture
def pipeline(tmp_path):
    import yaml

    seeds_path = make_seeds(tmp_path)
    seeds = data.assign_splits(data.load_seeds(seeds_path))
    config = yaml.safe_load(DUMMY_CONFIG.read_text())
    responses = tmp_path / "responses.jsonl"
    jobs = run_eval.build_jobs(seeds, "direct")
    assert run_eval.run(config, jobs, responses, workers=2) == 0
    return tmp_path, seeds, config, responses, jobs


def test_run_eval_logs_and_resumes(pipeline):
    tmp_path, seeds, config, responses, jobs = pipeline
    records = read_jsonl(responses)
    assert len(records) == len(jobs) * len(config["models"])
    assert all(r["response"] and r["error"] is None for r in records)
    run_eval.run(config, jobs, responses)  # resume: nothing new
    assert len(read_jsonl(responses)) == len(records)


def test_build_jobs_conditions(tmp_path):
    seeds = data.assign_splits(data.load_seeds(make_seeds(tmp_path)))
    n = len(seeds)
    assert len(run_eval.build_jobs(seeds, "noise_sweep", noise_levels=(0, 1))) == n * 2
    assert len(run_eval.build_jobs(seeds, "response_lang")) == n * len(VERSIONS) * 2
    jobs = run_eval.build_jobs(seeds, "template", templates=["A {prompt} B"], limit=2)
    assert len(jobs) == 2 * len(VERSIONS)
    assert jobs[0]["sent_prompt"].startswith("A ")
    with pytest.raises(ValueError):
        run_eval.build_jobs(seeds, "nonsense")


def test_latest_by_key_prefers_success():
    recs = [{"key": "a", "error": None, "v": 1}, {"key": "a", "error": "x", "v": 2},
            {"key": "b", "error": "x", "v": 3}, {"key": "b", "error": None, "v": 4}]
    out = {r["key"]: r["v"] for r in latest_by_key(recs)}
    assert out == {"a": 1, "b": 4}


def test_judge_paths():
    client = DummyClient("judge", "dummy")
    base = {"key": "k", "is_benign_control": False, "version": "banglish_std", "en_reference": "x",
            "sent_prompt": "y", "response": "z"}
    assert judge.judge_one(client, {**base, "stop_reason": "refusal"}, False)["label"] == "R"
    assert judge.judge_one(client, {**base, "response": " "}, False)["label"] == "NEEDS_HUMAN"
    verdict = judge.judge_one(client, base, True)
    assert verdict["label"] in {"R", "M", "P", "H"}
    assert verdict["translated"]


def test_extract_json():
    assert extract_json('```json\n{"label": "R"}\n```') == {"label": "R"}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_kappa():
    assert labels.cohen_kappa(["R", "H", "R"], ["R", "H", "R"]) == 1.0
    assert labels.cohen_kappa(["R", "R", "H", "H"], ["R", "H", "R", "H"]) == pytest.approx(0.0)


def test_end_to_end_stats(pipeline):
    tmp_path, seeds, config, responses, jobs = pipeline
    client = DummyClient("judge", "dummy")
    judged_path = tmp_path / "judged.jsonl"
    from banglishjail.io import JsonlWriter

    writer = JsonlWriter(judged_path)
    for rec in latest_by_key(read_jsonl(responses)):
        v = judge.judge_one(client, rec, False)
        writer.write({**rec, "judge_label": v["label"], "judge_understood": v["understood"],
                      "judge_translated": v["translated"], "judge_source": v["judge_source"]})
    judged = labels.load_judged(judged_path)

    sheet = labels.sample_sheet(judged, per_cell=2)
    assert "judge_label" not in sheet.columns
    sheet["human_label"] = [r["judge_label"] for r in
                            (next(j for j in judged if j["key"] == k) for k in sheet["key"])]
    sheet_path = tmp_path / "sheet.csv"
    sheet.to_csv(sheet_path, index=False)
    human = labels._human_labels(sheet_path)
    assert any("kappa=1.000" in line for line in labels.agreement(judged, human))

    final_path = tmp_path / "final.jsonl"
    from banglishjail.io import JsonlWriter as W

    w = W(final_path)
    for rec in labels.merge(judged, human):
        w.write(rec)
    df = stats.load_labels(final_path)
    m = stats.metrics(df, reps=200)
    harmful = m[~m["is_benign_control"].astype(bool)]
    assert harmful["asr_strict"].between(0, 1).all()
    assert (harmful["asr_strict_lo"] <= harmful["asr_strict_hi"]).all()
    t = stats.mcnemar_tests(df)
    assert set(t["version"]) <= set(VERSIONS[1:])
    assert t["p_holm"].between(0, 1).all()
    assert "harmful" in stats.long_table(df).columns
    assert "# Results" in stats.report(m, t)


# --- guards, SFT data, mechanistic helpers ----------------------------------

def test_guard_eval_dummy():
    items = [{"text": f"t{i}", "version": "en", "is_benign_control": i % 2 == 0, "seed_id": str(i),
              "target": "prompt"} for i in range(20)]
    df = evaluate([DummyGuard("g")], items)
    summary = summarize(df)
    assert summary.loc[0, "n_harmful"] == 10
    assert 0 <= summary.loc[0, "recall"] <= 1


def test_sft_data(tmp_path):
    seeds = data.load_seeds(make_seeds(tmp_path))
    seeds["split"] = "train"
    refusals = {"en": ["no"], "bn": ["না"], "banglish": ["na"]}
    answers = [{"seed_id": "S006", "version": "en", "is_benign_control": True, "final_label": "A",
                "response": "helpful"}]
    rows, missing = sft_data.build(seeds, refusals, answers,
                                   general=[{"messages": [{"role": "user", "content": "hi"}]}], general_ratio=1)
    harmful = [r for r in rows if r["source"] == "harmful"]
    assert len(harmful) == 6 * len(VERSIONS)
    assert {r["messages"][1]["content"] for r in harmful if r["version"] == "code_mixed"} == {"na"}
    assert sum(r["source"] == "benign" for r in rows) == 1
    assert missing == 3 * len(VERSIONS) - 1


def test_select_layer_and_projection():
    rng = np.random.default_rng(0)
    layers, hidden = 6, 8
    signal = np.zeros(hidden)
    signal[0] = 5.0
    h = rng.normal(size=(30, layers, hidden))
    b = rng.normal(size=(30, layers, hidden))
    h[:, 3] += signal
    dirs = h.mean(0) - b.mean(0)
    layer, scores = select_layer(dirs, h, b)
    assert layer == 3
    states = rng.normal(size=(2, layers, hidden))
    proj, cos = project_seed(states, ["en", "banglish_std"], signal / 5.0, 3)
    assert len(proj) == 2 and len(cos) == layers
