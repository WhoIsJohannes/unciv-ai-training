# v2 Acceptance Analysis

## AC1 — attributable curves

- Tiny `v1-reinforce`: 12 rounds, final win-rate 50.0% (n=108)
- Tiny `blind-critic`: 12 rounds, final win-rate 31.5% (n=108)
- Tiny `rich-critic`: 12 rounds, final win-rate 53.7% (n=108)

Overlay plot: `/Users/j/Unciv-onnx-selfplay-v2/training-runs/v2/curve_tiny_overlay.png`

## AC2 — convergence (does the critic steady the curve?)

- last-4 win-rate stddev: v1-reinforce=22.74pp, blind-critic=11.15pp → blind-critic STEADIER ✓
- last-4 mean win-rate: v1-reinforce=49.8%, blind-critic=35.0%

## AC3 — ceiling (does seeing the board help on Medium?)

- Medium final eval (n=200): blind-critic=28.9%, rich-critic=14.7%
- two-proportion z=-3.48, one-sided p(rich>blind)=0.9997 → NOT significant at p<0.05 (reported plainly)

Medium overlay plot: `/Users/j/Unciv-onnx-selfplay-v2/training-runs/v2/curve_medium_overlay.png`

