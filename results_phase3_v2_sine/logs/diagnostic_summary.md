# Gate Diagnostics

## Gate Status
- E1 gate failed: FedProTrack trails IFCA by 0.1115 re-ID.
- E4 gate failed: FedProTrack has zero non-dominated budget points.

## Default Budget Points
- CompressedFedAvg fe=1: bytes=3800.0, auc=10.0624
- CompressedFedAvg fe=2: bytes=1800.0, auc=10.2687
- CompressedFedAvg fe=5: bytes=600.0, auc=10.5949
- CompressedFedAvg fe=10: bytes=200.0, auc=10.6961
- FedAvg-Full fe=1: bytes=4560.0, auc=10.3102
- FedAvg-Full fe=2: bytes=2160.0, auc=10.5477
- FedAvg-Full fe=5: bytes=720.0, auc=10.7371
- FedAvg-Full fe=10: bytes=240.0, auc=10.7263
- FedDrift fe=1: bytes=4560.0, auc=10.7690
- FedDrift fe=2: bytes=2160.0, auc=10.7559
- FedDrift fe=5: bytes=720.0, auc=10.7735
- FedDrift fe=10: bytes=240.0, auc=10.7623
- FedProTrack fe=1: bytes=12000.0, auc=10.5334
- FedProTrack fe=2: bytes=6000.0, auc=10.3286
- FedProTrack fe=5: bytes=2400.0, auc=10.3724
- FedProTrack fe=10: bytes=1200.0, auc=10.3014
- FedProto fe=1: bytes=6080.0, auc=9.6060
- FedProto fe=2: bytes=2880.0, auc=9.4847
- FedProto fe=5: bytes=960.0, auc=9.2535
- FedProto fe=10: bytes=320.0, auc=9.3843
- Flash fe=1: bytes=4560.0, auc=10.3102
- Flash fe=2: bytes=2160.0, auc=10.5149
- Flash fe=5: bytes=720.0, auc=10.6168
- Flash fe=10: bytes=240.0, auc=10.5909
- IFCA fe=1: bytes=9120.0, auc=10.7267
- IFCA fe=2: bytes=4320.0, auc=10.7313
- IFCA fe=5: bytes=1440.0, auc=10.7414
- IFCA fe=10: bytes=480.0, auc=10.7371
- TrackedSummary fe=1: bytes=10640.0, auc=10.3102
- TrackedSummary fe=2: bytes=5040.0, auc=10.5477
- TrackedSummary fe=5: bytes=1680.0, auc=10.7371
- TrackedSummary fe=10: bytes=560.0, auc=10.7263

## Alpha Sweep Diagnostics
- alpha=0.0
  FedProTrack: final_acc=0.651, auc=12.028, dip=0.202, recovery=2.000
  IFCA: final_acc=0.663, auc=12.186, dip=0.190, recovery=1.200
  FedDrift: final_acc=0.663, auc=12.222, dip=0.220, recovery=1.000
- alpha=0.25
  FedProTrack: final_acc=0.579, auc=11.059, dip=0.128, recovery=4.800
  IFCA: final_acc=0.586, auc=11.118, dip=0.141, recovery=4.800
  FedDrift: final_acc=0.582, auc=11.143, dip=0.131, recovery=5.200
- alpha=0.5
  FedProTrack: final_acc=0.540, auc=10.483, dip=0.114, recovery=5.400
  IFCA: final_acc=0.541, auc=10.640, dip=0.131, recovery=4.800
  FedDrift: final_acc=0.536, auc=10.648, dip=0.133, recovery=4.800
- alpha=0.75
  FedProTrack: final_acc=0.555, auc=10.812, dip=0.114, recovery=5.800
  IFCA: final_acc=0.560, auc=11.062, dip=0.104, recovery=5.200
  FedDrift: final_acc=0.557, auc=11.073, dip=0.111, recovery=5.800
- alpha=1.0
  FedProTrack: final_acc=0.592, auc=11.368, dip=0.102, recovery=4.400
  IFCA: final_acc=0.608, auc=11.777, dip=0.109, recovery=4.400
  FedDrift: final_acc=0.606, auc=11.702, dip=0.113, recovery=3.800

## Module Ablations
- Full FedProTrack: reid=0.650, wrong_mem=0.350, entropy=0.569
- No temporal prior: reid=0.395, wrong_mem=0.605, entropy=0.061
- Hard assignment (omega=100): reid=0.650, wrong_mem=0.350, entropy=0.569
- No spawn/merge: reid=0.380, wrong_mem=0.620, entropy=0.000
- No post-spawn merge: reid=0.650, wrong_mem=0.350, entropy=0.569
- No sticky dampening: reid=0.650, wrong_mem=0.350, entropy=0.569
- No model-loss gate: reid=0.650, wrong_mem=0.350, entropy=0.569
- Phase A only (no aggregation): reid=0.380, wrong_mem=0.620, entropy=0.000
