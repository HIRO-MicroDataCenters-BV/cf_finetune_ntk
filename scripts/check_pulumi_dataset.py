#!/usr/bin/env python
"""QA for runs/pulumi/*.jsonl: mock-run every completion and apply the K8s policies.

No Pulumi CLI is needed. Each program is executed in a forked child process under
``pulumi.runtime.set_mocks`` (so ``import pulumi_kubernetes`` is paid once; ``set_mocks``
already registers the root Stack, so the program is driven with ``run_pulumi_func``), the
registered resources are captured from ``new_resource`` (keyed by type and name,
since a Deployment and its Service legitimately share a logical name) and handed to
``evaluate()`` from agentic-iac's ``policies.py`` (imported by path). A row passes
when the program executes, registers at least one workload container and has no
policy violations. Also checks statically that every ``k8s.`` reference is a
fully-qualified ``k8s.<group>.<version>.<Type>`` path (never ``k8s.types``).

Needs ``pulumi`` + ``pulumi_kubernetes`` importable. Either run it with such an
interpreter, or create a throwaway venv and point ``PULUMI_QA_VENV`` at it (the
script re-execs into that venv's python):

    uv venv /path/qa && uv pip install --python /path/qa/bin/python pulumi pulumi-kubernetes
    PULUMI_QA_VENV=/path/qa python scripts/check_pulumi_dataset.py [train.jsonl eval.jsonl]

Exit status is non-zero if any row fails.
"""

import argparse
import asyncio
import importlib.util
import json
import multiprocessing
import os
import re
import sys
import traceback
from pathlib import Path

QA_VENV = Path(os.environ["PULUMI_QA_VENV"]) if os.environ.get("PULUMI_QA_VENV") else None
POLICIES_PATH = Path("/home/ali/project/coge/agentic-iac/src/hiro_iac_cf/policies.py")
DEFAULT_FILES = ["runs/pulumi/train.jsonl", "runs/pulumi/eval.jsonl"]

FENCE = re.compile(r"^ ```python\n(.*)\n```$", re.S)
K8S_REF = re.compile(r"\bk8s\.[A-Za-z0-9_.]+")
K8S_OK = re.compile(r"^k8s\.(apps|core|meta|batch)\.v1\.[A-Z][A-Za-z0-9]+$")
SNAKE_KEY = re.compile(r"^[a-z]+(_[a-z0-9]+)+$")


def ensure_pulumi():
    """Re-exec into $PULUMI_QA_VENV when pulumi is not importable here; exit 2 with a hint otherwise."""
    try:
        import pulumi  # noqa: F401
    except ImportError:
        pass
    else:
        return
    venv_python = QA_VENV / "bin" / "python" if QA_VENV else None
    if venv_python and venv_python.exists() and Path(sys.prefix).resolve() != QA_VENV.resolve():
        os.execv(str(venv_python), [str(venv_python), *sys.argv])
    print(
        "pulumi is not importable. Create a QA venv and point PULUMI_QA_VENV at it:\n"
        "  uv venv <dir> && uv pip install --python <dir>/bin/python pulumi pulumi-kubernetes\n"
        "  PULUMI_QA_VENV=<dir> python scripts/check_pulumi_dataset.py",
        file=sys.stderr,
    )
    sys.exit(2)


def load_policies():
    spec = importlib.util.spec_from_file_location("hiro_policies", POLICIES_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_code(completion):
    m = FENCE.match(completion)
    if not m:
        raise ValueError("completion is not a single ' ```python ... ```' fenced block")
    return m.group(1)


def static_check(code):
    problems = []
    for ref in sorted(set(K8S_REF.findall(code))):
        if not K8S_OK.match(ref):
            problems.append(f"non-allow-listed reference {ref}")
    if "import pulumi\n" not in code or "import pulumi_kubernetes as k8s\n" not in code:
        problems.append("missing house-style imports")
    if code.count("pulumi.export(") != 1:
        problems.append("expected exactly one pulumi.export(...)")
    return problems


def camelize(obj):
    """Convert snake_case dict keys to camelCase (only if the mocks hand us snake_case)."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, str) and SNAKE_KEY.match(k):
                head, *rest = k.split("_")
                k = head + "".join(w.capitalize() for w in rest)
            out[k] = camelize(v)
        return out
    if isinstance(obj, list):
        return [camelize(v) for v in obj]
    return obj


def run_program(code):
    """Execute one program under Pulumi mocks; runs in a fresh forked child."""
    import pulumi
    from pulumi.runtime.stack import run_pulumi_func

    captured = {}

    class Mocks(pulumi.runtime.Mocks):
        def new_resource(self, args):
            if args.typ != "pulumi:pulumi:Stack":
                captured[f"{args.typ}::{args.name}"] = {"typ": args.typ, "inputs": args.inputs}
            return [f"{args.name}-id", args.inputs]

        def call(self, args):
            return {}

    pulumi.runtime.set_mocks(Mocks(), preview=False)

    def program():
        exec(compile(code, "__main__.py", "exec"), {"__name__": "__main__"})

    try:
        loop = asyncio.get_event_loop_policy().get_event_loop()
        loop.run_until_complete(run_pulumi_func(program))
    except Exception:  # noqa: BLE001 - reported per row
        return {"ok": False, "error": traceback.format_exc(limit=3), "resources": {}}
    resources = {}
    for name, r in captured.items():
        inputs = r["inputs"]
        if any(SNAKE_KEY.match(k) for k in inputs.get("metadata", {}) or {}) or any(
            SNAKE_KEY.match(k) for k in inputs.get("spec", {}) or {}
        ):
            inputs = camelize(inputs)
        resources[name] = inputs
    return {"ok": True, "resources": resources, "types": {n: r["typ"] for n, r in captured.items()}}


def main():
    ensure_pulumi()

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="*", default=DEFAULT_FILES)
    parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 2)
    parser.add_argument("-v", "--verbose", action="store_true", help="print every row, not just failures")
    args = parser.parse_args()

    import pulumi_kubernetes  # noqa: F401 - imported before forking so children inherit it

    policies = load_policies()
    rows = []
    for f in args.files:
        for i, line in enumerate(Path(f).read_text().splitlines()):
            rows.append((f, i, json.loads(line)))

    codes = []
    static_problems = {}
    for f, i, row in rows:
        try:
            code = extract_code(row["completion"])
        except ValueError as exc:
            static_problems[(f, i)] = [str(exc)]
            code = ""
        else:
            static_problems[(f, i)] = static_check(code)
        codes.append(code)

    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(processes=args.jobs, maxtasksperchild=1) as pool:
        results = pool.map(run_program, codes, chunksize=1)

    failures = 0
    per_file = {}
    for (f, i, row), result in zip(rows, results):
        kind = row.get("meta", {}).get("kind", "?")
        problems = list(static_problems[(f, i)])
        n_res = len(result["resources"])
        if not result["ok"]:
            problems.append("exception:\n" + result["error"].rstrip())
        else:
            containers = sum(len(policies._containers(r)) for r in result["resources"].values())
            if containers == 0:
                problems.append("no workload containers registered")
            problems.extend(policies.evaluate(result["resources"]))
        status = "OK" if not problems else "FAIL"
        per_file.setdefault(f, [0, 0])[0 if status == "OK" else 1] += 1
        if problems:
            failures += 1
        if problems or args.verbose:
            print(f"[{status}] {f}#{i} {kind} resources={n_res} " + " ".join(sorted(result.get("types", {}).values())))
            for p in problems:
                print("    - " + p.replace("\n", "\n      "))

    for f, (ok, bad) in per_file.items():
        print(f"{f}: {ok} passed, {bad} failed")
    print(f"QA {'PASS' if failures == 0 else 'FAIL'}: {len(rows) - failures}/{len(rows)} rows executed under mocks with zero policy violations")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
