import sys

sys.path.insert(0, "/home/foods/pro/pyct_old/pyct")
from algorithm.calibration.cal import Calibration

cal = Calibration(
    proj_path="/home/foods/pro/data/20260327-jz-1",
    dpixel=0.0748,
    num=6,
    w=1536,
    h=1944,
    bead_spacing=10.0,
)

result = cal.calculate()

print("=== 校准结果 ===")
print(f"SOD   : {result['SOD']:.4f} mm")
print(f"SDD   : {result['SDD']:.4f} mm")
print(f"u0    : {result['u0']:.4f} px")
print(f"v0    : {result['v0']:.4f} px")
print(f"theta : {result['theta']:.6f} rad")
print(f"eta   : {result['eta']:.6f} rad")
print()
print("=== Legacy 初值 ===")
for k, v in result["legacy"].items():
    print(f"  {k}: {v:.6f}")

print()
print("=== 与 legacy 基线对比 ===")
baseline = dict(SOD=907.3701, SDD=971.7076, u0=912.4626, v0=469.8098, theta=0.2708)
for k in ["SOD", "SDD", "u0", "v0", "theta"]:
    diff = result[k] - baseline[k]
    print(f"  {k}: new={result[k]:.4f}  baseline={baseline[k]:.4f}  diff={diff:+.4f}")
