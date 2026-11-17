"""CLI for the per-criterion robustness study.

  python -m research.run_robustness prepare            # build data/processed + splits.json
  python -m research.run_robustness extract            # features for all clean items
  python -m research.run_robustness fit                # fit SUT config, clean results table
  python -m research.run_robustness generate [--llm]   # adversarial variants
  python -m research.run_robustness validate export|auto|agree
  python -m research.run_robustness baselines finetune|llm
  python -m research.run_robustness eval               # attack results for all systems
  python -m research.run_robustness detect             # CDI / learned detector
  python -m research.run_robustness adaptive           # detector-aware attacks
  python -m research.run_robustness report             # tables + figures
"""

import argparse
import sys

from research.robustness import common  # noqa: F401  (sets HF cache env before model imports)


def cmd_prepare(_):
    from research.robustness import data
    data.build_all()
    print("wrote", common.DATA_PROCESSED)


def cmd_extract(args):
    from research.robustness import data
    from research.robustness.features import FeatureStore
    store = FeatureStore()
    for name in args.datasets:
        items = data.load(name)
        rows = [{"question": it.question, "reference": it.reference, "student": it.student} for it in items]
        store.ensure(rows, desc=name)
        print(f"{name}: {len(items)} items cached")


def cmd_fit(args):
    from research.robustness import pipeline
    pipeline.fit_and_report_clean(args.datasets)


def cmd_generate(args):
    from research.robustness import attacks
    attacks.generate_all(args.datasets, use_llm=args.llm, n_per_attack=args.n)


def cmd_validate(args):
    from research.robustness import validation
    getattr(validation, args.action)()


def cmd_baselines(args):
    from research.robustness import systems
    if args.kind == "finetune":
        systems.run_finetune_all(args.datasets, seeds=args.seeds)
    else:
        systems.run_llm_all(args.datasets, mode=args.mode, limit=args.limit)


def cmd_eval(args):
    from research.robustness import pipeline
    pipeline.evaluate_attacks(args.datasets)


def cmd_detect(args):
    from research.robustness import detector
    detector.run(args.datasets)


def cmd_adaptive(args):
    from research.robustness import adaptive
    adaptive.run(args.datasets, use_llm=args.llm)


def cmd_report(args):
    from research.robustness import report
    report.build(args.datasets)


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn):
        sp = sub.add_parser(name)
        sp.add_argument("--datasets", nargs="+", default=common.DATASETS)
        sp.set_defaults(fn=fn)
        return sp

    add("prepare", cmd_prepare)
    add("extract", cmd_extract)
    add("fit", cmd_fit)
    g = add("generate", cmd_generate)
    g.add_argument("--llm", action="store_true", help="also run LLM-based attacks (A2, A4)")
    g.add_argument("--n", type=int, default=300, help="max source answers per attack per split")
    v = add("validate", cmd_validate)
    v.add_argument("action", choices=["export", "auto", "agree"])
    b = add("baselines", cmd_baselines)
    b.add_argument("kind", choices=["finetune", "llm"])
    b.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 87])
    b.add_argument("--mode", choices=["holistic", "analytic", "both"], default="both")
    b.add_argument("--limit", type=int, default=None)
    add("eval", cmd_eval)
    add("detect", cmd_detect)
    a = add("adaptive", cmd_adaptive)
    a.add_argument("--llm", action="store_true")
    add("report", cmd_report)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
