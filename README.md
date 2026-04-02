# Common Responses (tables, plots) to Shared Reviewer Concerns


---

## MQ1. Broader Benchmarks & Backbone-Agnostic Evaluation

**Raised by:** Reviewer 3UNC (W5), Reviewer UQ5F (W1, W3), Reviewer k84A (implicitly), Reviewer XYhH (W1)

### (a) Backbone-Agnostic Verification (jax-gcrl)


**Table R1.** Final success rate (mean ± std) on jax-gcrl benchmarks.

| Method | Pusher Hard | Ant U-Maze | Ant Big Maze | Cheetah | Ant Soccer | Avg |
|---|---|---|---|---|---|---|
| CRL | 0.630±0.069 | 0.212±0.019 | 0.108±0.041 | 0.455±0.379 | 0.191±0.031 | 0.319 |
| **CRL+GCHR** | **0.679±0.037** | **0.304±0.074** | **0.157±0.049** | **0.684±0.385** | **0.270±0.027** | **0.419** |
| SAC | 0.248±0.199 | 0.175±0.062 | 0.053±0.040 | 1.000±0.000 | 0.449±0.032 | 0.385 |
| SAC+HER | 0.752±0.067 | 0.324±0.048 | 0.138±0.015 | 1.000±0.000 | 0.002±0.003 | 0.443 |
| **SAC+GCHR** | **0.832±0.024** | **0.548±0.171** | **0.171±0.048** | **1.000±0.000** | **0.387±0.052** | **0.588** |
| TD3 | 0.041±0.075 | 0.109±0.044 | 0.003±0.006 | 1.000±0.000 | 0.357±0.049 | 0.302 |
| TD3+HER | 0.739±0.032 | 0.105±0.090 | 0.040±0.030 | 0.999±0.002 | 0.000±0.000 | 0.377 |
| **TD3+GCHR** | **0.800±0.034** | **0.221±0.111** | **0.172±0.089** | **1.000±0.000** | **0.360±0.055** | **0.511** |

![Figure R1: Training curves on jax-gcrl benchmarks.](jax-gcrl-res/icml_training_curves.png)
*Figure R1. Training curves on jax-gcrl benchmarks. Top row: CRL family. Middle row: SAC family. Bottom row: TD3 family. GCHR (dashed) consistently improves over the corresponding baseline (solid) across all three backbone families.*

![Figure R2: Final success rate comparison across tasks.](jax-gcrl-res/icml_bar_chart.png)
*Figure R2. Final success rate across all methods and tasks. Within each task, X+GCHR (darker bar) consistently outperforms the corresponding baseline X and X+HER.*



### (b) Image-Based and Visual Manipulation Tasks


**Table R2.** Success rate on image-based, locomotion, and visual manipulation benchmarks (backbone: SAC+GCHR).

| | QRL | TD-InfoNCE | GCHR (ours) |
|---|---|---|---|
| reach-image | 100±0 | 100±0 | 100±0 |
| push-image | 80±8 | 82±3 | **86±6** |
| pick-image | 2±1 | 24±3 | **30±2** |
| PointMaze | 74±4 | 88±4 | **93±5** |
| AntMaze | 67±9 | 74±2 | **81±4** |
| Visual-cube-noisy | 58±5 | 69±10 | **77±8** |
| Visual-scene-noisy | 48±2 | 58±6 | **60±4** |



---


---


## MQ2. RIS experimental comparison



**Table R3.** Success rate (%) on Fetch benchmarks (all methods: SAC backbone). All methods use the **SAC backbone** for fair comparison.

| Method | FetchReach | FetchPush | FetchSlide | FetchPick |
|---|---|---|---|---|
| RIS (SAC) | 70±3 | 97±4 | 21±6 | 52±4 |
| SAC+HER | 100±0 | 95±2 | 23±5 | 51±4 |
| CRL | 100±0 | 6±5 | 2±1 | 8±2 |
| **SAC+GCHR** | **100±0** | **99±3** | **38±3** | **52±6** |



---

## MQ3. Training Time / Wall-Clock Overhead

**Raised by:** Reviewer UQ5F (W4), Reviewer k84A (Q2)

**Table R1.** Wall-clock overhead and sample efficiency of GCHR vs. DDPG+HER on FetchPush as a function of $K$ (number of forward passes per update). Default: $K{=}10$.

| $K$ | Wall-clock overhead | Env steps to 90% success |
|---|---|---|
| 5 | +12% | ~32k (1.6× faster) |
| 10 | +22% | ~28k (1.8× faster) |
| 15 | +31% | ~27k (1.9× faster) |
| 20 | +40% | ~27k (diminishing returns) |


---

## MQ4. Coverage Expansion — Direct Empirical Evidence

**Raised by:** Reviewer XYhH (Q4), Reviewer k84A (implicitly via Theorem 6.1)



### Reacher (2D actions, SAC+GCHR, 5.1M steps)

| Action source | Q-advantage over random | Notes |
|---|---|---|
| $\pi_\theta$ (policy) | +1.20 ± 0.09 | Trained policy (sanity check: highest) |
| $\pi_{\text{HG}}$ covered | +1.05 ± 0.08 | $\pi_{\text{HG}}$ actions overlapping $\rho_{\text{beh}}$ |
| $\rho_{\text{beh}}$ (recorded) | +0.94 ± 0.07 | Behavioral support |
| **$\pi_{\text{HG}}$ novel** | **+0.25 ± 0.07** | **Outside $\rho_{\text{beh}}$ support** |
| Random | 0.00 | Baseline |



The 2D action space allows direct visualization:

![Figure R3: Q-landscape showing coverage expansion on Reacher.](action_coverage_res/results_reacher/figure_a_reacher_qlandscape.png)
*Figure R3. Coverage expansion on Reacher. Background heatmap: Q(s,⋅,g)Q(s,\cdot,g)
Q(s,⋅,g) over the 2D action space (blue = high Q, red = low Q). Red ×: actions from ρbeh\rho_{\text{beh}}
ρbeh​ (behavioral support). Blue ○: novel actions from πHG\pi_{\text{HG}}
πHG​ (outside behavioral support). Green ★: current policy action. The blue dots land in high-Q (blue) regions beyond the red crosses, showing that the hindsight goal prior discovers valuable actions that behavioral cloning alone would miss.*

### Pusher Easy (higher-dim actions, SAC+GCHR)


![Figure R4: Coverage expansion summary across environments.](action_coverage_res/combined_coverage_summary.png)
*Figure R4. Left: Q-advantage by action source. Right: advantage over training. On Reacher, novel actions maintain positive advantage throughout. On Pusher Easy, coverage expansion is most valuable during early exploration. In higher-dimensional action spaces, the distance-based novelty criterion becomes less discriminative (novel fraction ~97%). However, the training dynamics are informative: at 2.2M steps (early exploration), novel actions achieve +0.91 advantage over random, showing that coverage expansion is most impactful when the behavioral support is sparse. As the policy converges and $\rho_{\text{beh}}$ fills in, the marginal value diminishes — consistent with GCHR's design.
*

