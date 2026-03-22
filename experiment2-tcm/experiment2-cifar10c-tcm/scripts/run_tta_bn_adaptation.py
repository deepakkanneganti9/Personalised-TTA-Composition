import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from tta_techniques.tta_bn_adapter import adapt_client_with_tta_bn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fl-run-dir", required=True)
    parser.add_argument("--client-name", required=True)
    parser.add_argument("--corruption", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--split", default="test")
    parser.add_argument("--severity", type=int, default=1)
    parser.add_argument("--cifar10-c-root", default="Data/CIFAR-10-C")
    parser.add_argument("--allowed-classes", nargs="*", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pseudo-sample-size", type=float, default=32.0)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--num-threads", type=int, default=1)
    args = parser.parse_args()

    summary = adapt_client_with_tta_bn(
        fl_run_dir=Path(args.fl_run_dir).resolve(),
        client_name=args.client_name,
        corruption=args.corruption,
        output_dir=Path(args.output_dir).resolve(),
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        cifar10_c_root=Path(args.cifar10_c_root).resolve(),
        split=args.split,
        severity=args.severity,
        allowed_classes=args.allowed_classes,
        seed=args.seed,
        pseudo_sample_size=args.pseudo_sample_size,
        log_every=args.log_every,
        num_threads=args.num_threads,
    )
    print(summary)


if __name__ == "__main__":
    main()
