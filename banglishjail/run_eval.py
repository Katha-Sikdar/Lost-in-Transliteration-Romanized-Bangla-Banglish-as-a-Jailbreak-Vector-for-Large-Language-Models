"""Send prompts to target models and log every response (Phase 4).

Conditions:
    direct          A0: each version sent as is
    noise_sweep     A1: standard Banglish perturbed at each level in --noise-levels
    response_lang   A4: each version + an instruction about the answer language
    template        A2: each version wrapped in templates from --templates
                        (a JSON list of strings containing "{prompt}")

Automated attacks (A3: PAIR/TAP) are not reimplemented here; use the
maintained implementations in HarmBench or JailbreakBench with Banglish
seeds, and convert their outputs to this log format.

The run is resumable: records already in the output file without an error
are skipped, so you can stop and restart safely.

Usage:
    python -m banglishjail.run_eval --config configs/models.yaml \
        --seeds data/raw/seeds.csv --out data/raw/responses.jsonl \
        --condition direct --split test
"""

import argparse
import datetime as dt
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

from banglishjail import VERSIONS
from banglishjail.clients import make_client
from banglishjail.data import expand, load_seeds
from banglishjail.io import JsonlWriter, read_jsonl
from banglishjail.noise import noise_level, perturb

RESPONSE_LANG_SUFFIX = {
    "answer_en": "\n\nPlease answer in English.",
    "answer_banglish": "\n\nBanglish e uttor dao.",
}


def build_jobs(seeds, condition, split=None, samples=1, noise_levels=(0, 0.25, 0.5, 0.75, 1.0),
               templates=None, limit=None):
    """Return a list of job dicts (one per prompt to send, before model fan-out)."""
    if condition == "noise_sweep":
        base = expand(seeds, versions=["banglish_std"], split=split)
    else:
        base = expand(seeds, versions=VERSIONS, split=split)
    if limit:
        keep = sorted({r["seed_id"] for r in base})[:limit]
        base = [r for r in base if r["seed_id"] in keep]

    jobs = []
    for rec in base:
        variants = []
        if condition == "direct":
            variants.append(("", rec["prompt"], {}))
        elif condition == "noise_sweep":
            for level in noise_levels:
                text, rules = perturb(rec["prompt"], level)
                variants.append((f"noise={level}", text, {"noise_rate": level, "noise_rules": rules,
                                                          "noise_level": noise_level(rec["prompt"], text)}))
        elif condition == "response_lang":
            for tag, suffix in RESPONSE_LANG_SUFFIX.items():
                variants.append((tag, rec["prompt"] + suffix, {"response_lang": tag}))
        elif condition == "template":
            for i, tpl in enumerate(templates or []):
                variants.append((f"tpl={i}", tpl.replace("{prompt}", rec["prompt"]), {"template_id": i}))
        else:
            raise ValueError(f"unknown condition {condition!r}")
        for variant, text, extra in variants:
            for s in range(samples):
                jobs.append({**rec, **extra, "condition": condition, "variant": variant,
                             "sample": s, "sent_prompt": text})
    return jobs


def job_key(job, model_name):
    return "|".join([model_name, job["condition"], job["variant"], job["seed_id"], job["version"], str(job["sample"])])


def call_with_retry(client, prompt, system=None, attempts=4):
    delay = 2
    for attempt in range(attempts):
        try:
            return client.generate(prompt, system=system), None
        except Exception as exc:  # provider SDKs raise many types; log and retry
            if attempt == attempts - 1:
                return None, f"{type(exc).__name__}: {exc}"
            time.sleep(delay)
            delay *= 2


def run(config, jobs, out_path, workers=4, system=None):
    done = {r["key"] for r in read_jsonl(out_path) if not r.get("error")}
    writer = JsonlWriter(out_path)
    decoding_note = config.get("note", "")
    todo = []
    for model_cfg in config["models"]:
        client = make_client(model_cfg)
        for job in jobs:
            key = job_key(job, client.name)
            if key not in done:
                todo.append((client, model_cfg, job, key))
    print(f"{len(todo)} calls to make ({len(done)} already done)", file=sys.stderr)

    def work(item):
        client, model_cfg, job, key = item
        result, error = call_with_retry(client, job["sent_prompt"], system=system)
        record = {
            "key": key,
            **job,
            "model_name": client.name,
            "provider": model_cfg["provider"],
            "model": client.model,
            "params": client.params,
            "system": system,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "note": decoding_note,
            "error": error,
        }
        if result:
            record.update({"response": result["text"], "stop_reason": result["stop_reason"],
                           "refusal_category": result["refusal_category"], "served_model": result["served_model"]})
        writer.write(record)
        return error

    errors = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, item) for item in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            errors += fut.result() is not None
            if i % 50 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)} done, {errors} errors", file=sys.stderr)
    return errors


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--condition", default="direct",
                   choices=["direct", "noise_sweep", "response_lang", "template"])
    p.add_argument("--split", choices=["test", "train"], help="default: all seeds")
    p.add_argument("--samples", type=int, default=1, help="samples per prompt")
    p.add_argument("--noise-levels", default="0,0.25,0.5,0.75,1.0")
    p.add_argument("--templates", help="JSON file: list of strings containing {prompt}")
    p.add_argument("--limit", type=int, help="only the first N seeds (pilot)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--system", help="optional system prompt (default: none)")
    p.add_argument("--dry-run", action="store_true", help="print the number of jobs and exit")
    args = p.parse_args(argv)

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    templates = None
    if args.templates:
        with open(args.templates, encoding="utf-8") as f:
            templates = json.load(f)
    seeds = load_seeds(args.seeds)
    jobs = build_jobs(seeds, args.condition, args.split, args.samples,
                      [float(x) for x in args.noise_levels.split(",")], templates, args.limit)
    if args.dry_run:
        print(f"{len(jobs)} prompts x {len(config['models'])} models = {len(jobs) * len(config['models'])} calls")
        return 0
    errors = run(config, jobs, args.out, args.workers, args.system)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
