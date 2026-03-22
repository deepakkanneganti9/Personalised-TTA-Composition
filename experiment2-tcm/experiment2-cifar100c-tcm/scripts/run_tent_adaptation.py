import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from tta_techniques.tent_adapter import adapt_client_with_tent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fl-run-dir", required=True)
    parser.add_argument("--client-name", required=True)
    parser.add_argument("--corruption", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--split", default="test")
    parser.add_argument("--severity", type=int, default=1)
    parser.add_argument("--cifar-c-root", dest="cifar_c_root", default="../CIFAR100/Data/CIFAR-100-C")
    parser.add_argument("--cifar100-c-root", dest="cifar_c_root")
    parser.add_argument("--allowed-classes", nargs="*", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = adapt_client_with_tent(
        fl_run_dir=Path(args.fl_run_dir).resolve(),
        client_name=args.client_name,
        corruption=args.corruption,
        output_dir=Path(args.output_dir).resolve(),
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_batches=args.max_batches,
        cifar_c_root=Path(args.cifar_c_root).resolve(),
        split=args.split,
        severity=args.severity,
        allowed_classes=args.allowed_classes,
        seed=args.seed,
    )
    print(summary)


if __name__ == "__main__":
    main()
