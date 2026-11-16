#!/usr/bin/env python3
"""Run a jimm script (validate.py, benchmark.py, ...) once per model.

Mirrors timm's ``bulk_runner.py``. Each child script must print a JSON summary
after a line containing ``--result``.

Examples:
    python bulk_runner.py --model-list 'resnet*' --results-file bench.csv \
        benchmark.py --amp -b 64
    python bulk_runner.py --model-list all --results-file val.csv \
        validate.py --data-dir /data/imagenet --checkpoint-dir ./checkpoints
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time

from jimm import is_model, list_models


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a script once per jimm model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-list", default="", help="wildcard, 'all', or file of names")
    parser.add_argument("--results-file", default="", help="output CSV path")
    parser.add_argument("--sort-key", default="", help="column used to sort results")
    parser.add_argument("--pretrained", action="store_true", help="only models with weights")
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between invocations")
    parser.add_argument("--no-python", action="store_true", help="execute the script directly")
    parser.add_argument("-m", "--module", action="store_true", help="run the script as a module")
    parser.add_argument("script", help="script to launch for each model")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def _model_names(args):
    if args.pretrained:
        raise SystemExit("jimm registers no pretrained weights; drop --pretrained")
    if args.model_list in ("all", "all_in1k", "all_res"):
        return list_models()
    if os.path.isfile(args.model_list):
        with open(args.model_list) as file:
            names = [line.strip() for line in file if line.strip()]
        if not names:
            raise SystemExit(f"model list is empty: {args.model_list}")
        return names
    if is_model(args.model_list):
        return [args.model_list]
    names = list_models(args.model_list)
    if not names:
        raise SystemExit(f"no models match {args.model_list!r}")
    return names


def _command(args, model):
    if args.no_python:
        if args.module:
            raise SystemExit("--no-python and --module cannot be combined")
        command = [args.script]
    else:
        command = [os.environ.get("PYTHON_EXEC", sys.executable), "-u"]
        if args.module:
            command.append("-m")
        command.append(args.script)
    return [*command, *args.script_args, "--model", model]


def _parse_result(output):
    marker = "--result"
    if marker not in output:
        raise ValueError("child script did not print a --result summary")
    text = output.split(marker)[-1].lstrip()
    # Warnings emitted during interpreter shutdown can follow the JSON value.
    return json.JSONDecoder().raw_decode(text)[0]


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    args = _parse_args(argv)
    names = _model_names(args)
    default_key = "infer_samples_per_sec" if "benchmark" in args.script else "top1"
    sort_key = args.sort_key or default_key
    results, errors = [], []
    print(f"Running {args.script} on {len(names)} models (sort key: {sort_key})")
    for model in names:
        try:
            output = subprocess.check_output(_command(args, model), stderr=subprocess.STDOUT)
            result = _parse_result(output.decode("utf-8"))
            results.append(
                result if isinstance(result, dict) else {"result": result, "model": model}
            )
        except Exception as error:
            errors.append({"model": model, "error": str(error)})
            print(f"  {model}: {error}")
        if args.delay:
            time.sleep(args.delay)

    if results and sort_key in results[0]:
        results.sort(key=lambda row: row[sort_key], reverse=True)
    if args.results_file and results:
        _write_csv(args.results_file, results)
        print(f"Wrote {args.results_file}")
    if errors:
        print(f"{len(errors)} models failed")
    print(f"--result\n{json.dumps(results, indent=4)}")
    return results


if __name__ == "__main__":
    main()
