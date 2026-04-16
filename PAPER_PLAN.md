# Paper Plan: When Does Concept-Level Aggregation Help? A Crossover Law for Federated Learning under Concept Drift

**Venue**: NeurIPS 2026
**Page limit**: 9 pages (main body) + unlimited references + appendix
**Style**: neurips_2026.sty

---

## Claims-Evidence Matrix

| # | Claim | Evidence | Section |
|---|-------|----------|---------|
| C1 | Concept-level aggregation beats global iff SNR > C-1 | Theorem 1 + 91.7% alignment across 108 configs | §3, §5 |
| C2 | The crossover is fundamental (minimax lower bound) | Theorem 4 proof | §3 |
| C3 | James-Stein shrinkage adaptively interpolates and dominates both extremes | Theorem 3 + best in 62% of configs, beats both in 72% | §4, §5 |
| C4 | Practical clustering methods (IFCA, CFL) are suboptimal in OLS regime | 0/108 wins for IFCA and CFL | §5 |
| C5 | Theory predicts real-data outcomes on CIFAR-100 | Multiclass OVR B_j^2 bridge, SNR 7-11 > C-1=3 | §5 |
| C6 | Non-stationary drift affects the crossover via effective sample size | Corollary (Theorem 2) + regime analysis | §3, §5 |

---

## Section Plan

### §1 Introduction (~1.5 pages)

**Story arc**: Federated learning under concept drift faces a fundamental design choice — aggregate globally (more data, interference bias) or per-concept (no bias, higher variance). Prior work proposes heuristic solutions (IFCA, CFL, FedDrift) without characterizing *when* each strategy wins.

**Key paragraphs**:
1. FL + concept drift motivation (2-3 sentences)
2. The aggregation granularity question (the central puzzle)
3. Preview of results: crossover law SNR > C-1, shrinkage estimator, minimax lower bounds
4. Contributions bullet list (4 items)
5. Paper organization

**Citations**: McMahan et al. 2017, Ghosh et al. 2020 (IFCA), Sattler et al. 2021 (CFL), Jothimurugesan et al. 2023 (FedDrift), Panchal et al. 2023 (Flash), Chen et al. 2024 (FedCCFA), Deng et al. 2020 (APFL), Li et al. 2021 (Ditto), Xu et al. 2024 (ICML), Kim et al. 2024 (ICML)

### §2 Problem Setting (~1 page)

**Content**: Gaussian linear regression model, K clients, C concepts, concept assignment function, data model, OLS estimators (global vs concept-level), excess risk definition.

**Key equations**: Data model (y = <w*,x> + epsilon), global OLS, concept-level OLS, excess risk E_j.

**Source**: `docs/theory_framework.md` §1-2

### §3 Theory (~2.5 pages)

**§3.1 Bias-Variance Decomposition (Proposition 1)**
- Global estimator: interference bias + low variance
- Concept-level: no bias + high variance
- Source: `docs/theory_framework.md` §3

**§3.2 Crossover Condition (Theorem 1)**
- Main result: concept-level wins iff SNR_concept > C-1
- Interpretation: signal-to-noise ratio of concept separation
- Comparison with IFCA theory (Ghosh et al. 2020)
- Source: `docs/theory_framework.md` §4

**§3.3 Minimax Lower Bounds (Theorem 4)**
- No estimator avoids both costs simultaneously
- Confirms crossover is fundamental, not artifact of OLS
- Source: `docs/theory_framework.md` §7

**§3.4 Corollary: Non-Stationary Extension**
- Piecewise-stationary drift model
- Effective sample size N_eff(s) = (K/C)·n·min(s,W)
- Transient period s* after concept switch
- Demoted from standalone theorem to corollary per reviewer guidance
- Source: `docs/theory_framework.md` §5

### §4 Adaptive Shrinkage Estimator (~1 page)

**§4.1 James-Stein Shrinkage (Theorem 3)**
- Empirical Bayes framework
- Shrinkage coefficient lambda from between-concept dispersion
- Interpolates between global and concept-level
- Data-adaptive, no hyperparameter tuning
- Source: `docs/theory_framework.md` §6

**§4.2 Practical Implementation**
- Estimation of sigma_B^2 from client uploads
- Computational overhead: O(Cd) per round
- Connection to existing interpolation methods (APFL, Ditto)

### §5 Experiments (~2.5 pages)

**§5.1 Synthetic Gaussian Regression** (~1.5 pages)

*Setup*: K ∈ {10,20,40}, C ∈ {2,4,8}, delta ∈ {0.3,1.0,3.0,8.0}, tau ∈ {5,15,inf}, d=20, sigma=1.0, n=200, 3 seeds.

*Methods compared (6)*: FedAvg (global), Oracle (concept-level with true labels), Shrinkage (Theorem 3), IFCA (Ghosh 2020 analogue), CFL (Sattler 2021 analogue), APFL (Deng 2020 analogue, tuned alpha ∈ {0.2,0.5,0.8}).

*Key results*:
- Theory-experiment alignment: 91.7% (99/108)
- Shrinkage best in 62% of configs, beats both FedAvg+Oracle in 72%
- IFCA/CFL: 0 wins — clustering adds no value in balanced OLS
- APFL: 16 wins, mainly high-C stationary settings
- 9 mismatches all at crossover boundary

*Data source*: `tmp/crossover_6method_seedavg/summary.json`

**§5.2 CIFAR-100 Bridge** (~0.5 pages)

*Setup*: ResNet-18 features (d=128), 4 concepts via disjoint label split, 20 classes, K=10 clients.

*Method*: One-vs-rest OVR linear heads → B_j^2 = ||W_j - W_bar||_F^2

*Results*: SNR = 7.2-11.3 > C-1 = 3, Oracle acc 83-88% vs FedAvg 66-75%. Theory correctly predicts outcome.

*Data source*: `tmp/cifar100_bj_proxy_local/bj_proxy_results.json`

**§5.3 Regime Boundary Analysis** (~0.5 pages)

*Setup*: 8 configs varying K/C ∈ {1,2,5}, delta ∈ {0.5-3.0}, tau ∈ {5,10}.

*Results*: When SNR < C-1 (delta=0.5): FedAvg dominates 36/40 rounds. When SNR > C-1: Oracle dominates. Confirms Theorem 1 regime boundary.

*Data source*: `tmp/transient_analysis/transient_results.json`

### §6 Related Work (~0.5 pages)

- **Clustered FL**: IFCA (Ghosh 2020), CFL (Sattler 2021), FeSEM, FedRC — focus on cluster recovery, not when clustering helps
- **Personalized FL**: APFL (Deng 2020), Ditto (Li 2021), pFedMe — interpolation without principled crossover
- **FL under drift**: FedDrift (Jothimurugesan 2023), Flash (Panchal 2023), FedCCFA (Chen 2024) — heuristic adaptation
- **Theory**: Ghosh et al. 2020 convergence, Xu et al. 2024, Kim et al. 2024 — convergence rates, not when-to-cluster question

### §7 Conclusion (~0.3 pages)

Key message: The aggregation granularity question has a precise answer (SNR > C-1), the crossover is fundamental (minimax), and shrinkage provides a practical near-optimal strategy. Limitations: Gaussian/linear model, single real dataset.

### Appendix

- Full proofs (Proposition 1, Theorems 1, 3, 4, Corollary)
- Extended experimental results (full 108-config table)
- IFCA/CFL/APFL implementation details
- Additional CIFAR-100 analysis

---

## Figure Plan

| # | Type | Description | Data Source | Auto? |
|---|------|-------------|-------------|-------|
| F1 | Phase diagram | SNR vs C heatmap showing crossover boundary, with theory prediction overlay | `tmp/crossover_6method_seedavg/summary.json` | Yes |
| F2 | Bar chart | Best method distribution (6 methods, 108 configs) | `tmp/crossover_6method_seedavg/summary.json` | Yes |
| F3 | Scatter plot | Theory-predicted SNR vs empirical Oracle advantage, colored by match/mismatch | `tmp/crossover_6method_seedavg/summary.json` | Yes |
| F4 | Table | CIFAR-100 bridge results (3 seeds, B_j^2, SNR, accuracy) | `tmp/cifar100_bj_proxy_local/bj_proxy_results.json` | Yes |
| F5 | Line plot | Regime boundary: FedAvg vs Oracle dominance fraction across SNR values | `tmp/transient_analysis/transient_results.json` | Yes |
| F6 | Table | Main comparison table: 6 methods × key config subsets | `tmp/crossover_6method_seedavg/summary.json` | Yes |

---

## Citation Scaffolding

```bibtex
@inproceedings{mcmahan2017communication,
  title={Communication-efficient learning of deep networks from decentralized data},
  author={McMahan, Brendan and Moore, Eider and Ramage, Daniel and Hampson, Seth and Arcas, Blaise Ag{\"u}era y},
  booktitle={AISTATS},
  year={2017}
}

@inproceedings{ghosh2020efficient,
  title={An efficient framework for clustered federated learning},
  author={Ghosh, Avishek and Chung, Jichan and Yin, Dong and Ramchandran, Kannan},
  booktitle={NeurIPS},
  year={2020}
}

@inproceedings{sattler2021clustered,
  title={Clustered federated learning: Model-agnostic distributed multitask optimization under privacy constraints},
  author={Sattler, Felix and M{\"u}ller, Klaus-Robert and Samek, Wojciech},
  booktitle={IEEE TNNLS},
  year={2021}
}

@inproceedings{deng2020adaptive,
  title={Adaptive personalized federated learning},
  author={Deng, Yuyang and Kamani, Mohammad Mahdi and Mahdavi, Mehrdad},
  booktitle={arXiv:2003.13461},
  year={2020}
}

@inproceedings{li2021ditto,
  title={Ditto: Fair and robust federated learning through personalization},
  author={Li, Tian and Hu, Shengyuan and Beirami, Ahmad and Smith, Virginia},
  booktitle={ICML},
  year={2021}
}

@inproceedings{jothimurugesan2023federated,
  title={Federated learning under distributed concept drift},
  author={Jothimurugesan, Ellango and Hsieh, Kevin and Wang, Jianyu and Joshi, Gauri and Gibbons, Phillip B},
  booktitle={AISTATS},
  year={2023}
}

@inproceedings{panchal2023flash,
  title={Flash: Concept drift adaptation in federated learning},
  author={Panchal, Kunjal and Choudhary, Sunav and Mitra, Subrata and Mukherjee, Koyel and Sarkhel, Somdeb and Mitra, Saayan and Guan, Hui},
  booktitle={ICML},
  year={2023}
}

@inproceedings{chen2024fedccfa,
  title={Federated learning with concept drift adaptation},
  author={Chen, Hong-You and others},
  booktitle={NeurIPS},
  year={2024}
}

@inproceedings{li2020federated,
  title={Federated optimization in heterogeneous networks},
  author={Li, Tian and Sahu, Anit Kumar and Zaheer, Manzil and Sanjabi, Maziar and Talwalkar, Ameet and Smith, Virginia},
  booktitle={MLSys},
  year={2020}
}

@article{james1961estimation,
  title={Estimation with quadratic loss},
  author={James, William and Stein, Charles},
  journal={Proc. Fourth Berkeley Symp. Math. Statist. Prob.},
  year={1961}
}

@book{tsybakov2009introduction,
  title={Introduction to Nonparametric Estimation},
  author={Tsybakov, Alexandre B},
  publisher={Springer},
  year={2009}
}

@article{efron1973stein,
  title={Stein's estimation rule and its competitors — an empirical Bayes approach},
  author={Efron, Bradley and Morris, Carl},
  journal={JASA},
  year={1973}
}

@inproceedings{t2020personalized,
  title={Personalized federated learning with Moreau envelopes},
  author={T Dinh, Canh and Tran, Nguyen and Nguyen, Josh},
  booktitle={NeurIPS},
  year={2020}
}

@inproceedings{bartlett2020benign,
  title={Benign overfitting in linear regression},
  author={Bartlett, Peter L and Long, Philip M and Lugosi, G{\'a}bor and Tsigler, Alexander},
  journal={PNAS},
  year={2020}
}
```

---

## Formatting Notes

- NeurIPS uses `\usepackage{neurips_2026}` (anonymous submission: `\usepackage[preprint]{neurips_2026}`)
- 9-page main body limit (references and appendix unlimited)
- All theorems, propositions, corollaries use `\newtheorem`
- Math commands: `\E` for expectation, `\Var` for variance, `\norm{}` for norms
- Figures in `figures/` directory, tables inline
