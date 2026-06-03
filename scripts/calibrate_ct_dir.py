import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithm.calibration.cal import Calibration


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("proj_dir", type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    params = cfg["CalibrationParam"]
    calibration = Calibration(
        str(args.proj_dir),
        float(params["detectorPixelSize"]),
        int(params["BBNumber"]),
        int(params["detectorWidth"]),
        int(params["detectorHeight"]),
        params,
    )
    result = calibration.calculate_vshift_package()
    output = args.output or args.proj_dir / "calibration_vshift.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[Done] {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
