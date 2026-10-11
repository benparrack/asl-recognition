Word feature ablation (transformer, val split, participant-disjoint)

| features | channels | val top-1 | val top-5 | one-handed | two-handed | static | moderate | dynamic |
|---|---|---|---|---|---|---|---|---|
| dom_shape | 64 | 0.556 | 0.792 | 0.556 (n=13756) | n/a | 0.519 (n=4512) | 0.567 (n=4583) | 0.582 (n=4661) |
| both_shape | 128 | 0.564 | 0.794 | 0.564 (n=13756) | n/a | 0.535 (n=4512) | 0.565 (n=4583) | 0.590 (n=4661) |
| location | 139 | 0.614 | 0.823 | 0.614 (n=13756) | n/a | 0.590 (n=4512) | 0.613 (n=4583) | 0.638 (n=4661) |
| full | 153 | 0.639 | 0.843 | 0.639 (n=13756) | n/a | 0.611 (n=4512) | 0.644 (n=4583) | 0.661 (n=4661) |
