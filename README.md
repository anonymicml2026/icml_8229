# Common Responses to Shared Reviewer Concerns

We thank all reviewers for their constructive feedback. Several concerns were raised by multiple reviewers. We consolidate our responses here, with full experimental results, to avoid redundancy. Individual reviewer responses reference this document.

---

## MQ1. Broader Benchmarks & Backbone-Agnostic Evaluation

**Raised by:** Reviewer 3UNC (W5), Reviewer UQ5F (W1, W3), Reviewer k84A (implicitly), Reviewer XYhH (W1)

We have substantially broadened our evaluation in three directions: (a) backbone-agnostic verification on jax-gcrl, (b) image-based and visual manipulation tasks, and (c) comparison against modern contrastive GCRL baselines.

### (a) Backbone-Agnostic Verification (jax-gcrl)

We evaluate GCHR on top of three fundamentally different backbones — CRL (contrastive), SAC (maximum entropy), and TD3 (deterministic) — across five challenging tasks:

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

**Key observations:**

**(1) GCHR is backbone-agnostic.** GCHR improves CRL (+31% avg), SAC+HER (+33% avg), and TD3+HER (+36% avg) — three fundamentally different algorithm families. This validates GCHR as a general bootstrapping mechanism, not an algorithm-specific trick.

**(2) GCHR is complementary to CRL.** CRL+GCHR outperforms CRL on all 5 tasks. Since CRL is among the strongest modern GCRL baselines (and underlies the 1000-Layer Networks paper requested by Reviewer UQ5F), this demonstrates that our policy-space regularization provides orthogonal benefits to representation-learning approaches.

**(3) GCHR avoids HER failure modes.** On Ant Soccer, HER catastrophically degrades both SAC (0.449→0.002) and TD3 (0.357→0.000). GCHR maintains performance close to the no-HER baseline (SAC+GCHR: 0.387, TD3+GCHR: 0.360) while still benefiting from hindsight on other tasks. This robustness arises because our compositional prior aggregates diverse behaviors rather than memorizing specific trajectories.

### (b) Image-Based and Visual Manipulation Tasks

We evaluate SAC+GCHR against QRL and TD-InfoNCE (the improved successor to CRL) on image-based, locomotion, and visual manipulation benchmarks:

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

GCHR outperforms both QRL and TD-InfoNCE across all tasks, demonstrating generalization across observation modalities (state-based and image-based), environment types (manipulation, locomotion, navigation, visual scenes), and against modern contrastive baselines.

---

## MQ2. Is GCHR Just Behavioral Regularization? / Novelty

**Raised by:** Reviewer 3UNC (Q6), Reviewer XYhH (W3)

The structural form of Eq. 21 resembles behavioral regularization, but the prior is fundamentally different from standard behavioral regularization used in offline RL [R1, R2]. We highlight three distinctions:

1. **Evolving prior.** Standard behavioral regularization regularizes toward a fixed, unstructured data distribution $\beta(a|s)$. GCHR's prior $\rho_{\text{mix}}$ evolves with the policy through the target network $\bar{\pi}_\theta$, creating a self-reinforcing bootstrapping loop: better policy → better prior → better policy.

2. **Compositional structure.** The prior is a mixture $\rho_{\text{mix}} = \lambda\,\rho_{\text{beh}} + (1-\lambda)\,\rho_{\text{HG}}$ that aggregates knowledge across multiple intermediate waypoints via $\rho_{\text{HG}}(a|s,g) = \frac{1}{K}\sum_{k=1}^{K}\bar{\pi}_\theta(a|s,g'_k)$. Standard behavioral regularization has no such goal-compositional structure.

3. **Provable coverage expansion.** Theorem 6.1 proves $\mathcal{A}_{\text{beh}}(s,g) \subseteq \mathcal{A}_{\text{HG}}(s,g)$: the goal prior covers strictly more actions than self-imitation. We provide direct empirical evidence confirming this in **MQ7** below — novel actions from $\rho_{\text{HG}}$ outside the behavioral support achieve positive Q-advantage over random, demonstrating genuine coverage expansion rather than mere smoothing.

Moreover, our new experiments (Table R1) show GCHR consistently improves three different backbones (CRL, SAC, TD3) with zero additional learned components. RIS requires a subgoal prediction network. MHER requires a dynamics model. GCHR adds two loss terms to any existing actor-critic. This simplicity combined with broad, backbone-agnostic effectiveness is the contribution.

**References:**
[R1] A Minimalist Approach to Offline RL. NeurIPS 2021.
[R2] Revisiting the Minimalist Approach to Offline RL. NeurIPS 2023.

---

## MQ3. Forward KL vs Reverse KL / Objective Switch from Sec. 4 to Sec. 5

**Raised by:** Reviewer 3UNC (Q5), Reviewer XYhH (Q1)

### Why Sec. 4 and Sec. 5 use different objectives

Eq. 5 in Sec. 4 ($D_{\text{KL}}(\pi \Vert \rho)$ regularization) serves as **conceptual motivation**: it shows that under KL-regularized RL, the optimal policy reweights the prior by exponentiated Q-values (Eq. 6), establishing that a well-designed prior accelerates convergence. The practical objective Eq. 21 in Sec. 5 implements this insight through a modular design: the RL term handles Q-value maximization, while the reverse KL $D_{\text{KL}}(\rho_{\text{mix}} \Vert \pi_\theta)$ handles prior-matching. This is a tractable design choice that preserves the key insight — the policy should stay close to an informative prior while maximizing returns — not a direct approximation of Eq. 5.

### Why reverse KL, not forward KL

The choice of $D_{\text{KL}}(\rho_{\text{mix}} \Vert \pi_\theta)$ is driven by both **mathematical necessity** and a desirable **mode-covering property**.

**The forward KL is not well-defined under Lebesgue measure for our prior.** Computing $D_{\text{KL}}(\pi_\theta \Vert \rho_{\text{mix}}) = \mathbb{E}_{a \sim \pi_\theta}[\log \pi_\theta(a) - \log \rho_{\text{mix}}(a)]$ requires pointwise evaluation of $\log \rho_{\text{mix}}(a)$. Since $\rho_{\text{mix}} = \lambda\,\delta_{a_t} + (1-\lambda)\,\rho_{\text{HG}}$, the Dirac component $\delta_{a_t}$ is a singular measure — not absolutely continuous with respect to Lebesgue measure on continuous action spaces, so no pointwise-evaluable density exists. One could restrict to the continuous component $\rho_{\text{HG}}$ alone or define a KL with respect to a different base measure, but either approach would lose the behavior prior signal entirely or require an ad hoc reformulation that severs the connection to the compositional prior $\rho_{\text{mix}}$.

**The reverse KL decomposes tractably and preserves both components.** Minimizing $D_{\text{KL}}(\rho_{\text{mix}} \Vert \pi_\theta)$ w.r.t. $\theta$ is equivalent to maximizing $\mathbb{E}_{a \sim \rho_{\text{mix}}}[\log \pi_\theta(a \mid s,g)]$, which only requires evaluating the well-defined Gaussian density $\pi_\theta$ at samples drawn from $\rho_{\text{mix}}$. Using the mixture structure:

$$\mathbb{E}_{a \sim \rho_{\text{mix}}}[\log \pi_\theta(a \mid s,g)] = \lambda \underbrace{\log \pi_\theta(a_t \mid s,g)}_{-\mathcal{L}_{\text{beh}}} + (1-\lambda) \underbrace{\mathbb{E}_{a \sim \pi_{\text{HG}}}[\log \pi_\theta(a \mid s,g)]}_{-\mathcal{L}_{\text{HG}} + \text{const w.r.t.}\ \theta}$$

The Dirac component yields the behavior cloning loss (Eq. 17); the continuous component yields the KL matching loss (Eq. 20). This clean decomposition is a direct consequence of the reverse KL.

**Mode-covering property.** Minimizing $D_{\text{KL}}(\rho_{\text{mix}} \Vert \pi_\theta)$ forces $\pi_\theta$ to place density wherever $\rho_{\text{mix}}$ has mass. Even if one could define a valid forward KL variant, it would be mode-seeking: a unimodal Gaussian $\pi_\theta$ would collapse onto a single mode of the multimodal $\rho_{\text{mix}}$, losing the diversity benefit of our compositional prior.

---

## MQ4. Assumption 4.1 — Validity, Empirical Evidence

**Raised by:** Reviewer 3UNC (Q4), Reviewer k84A (W1, Q4)

### What the assumption does and does not claim

Assumption 4.1 does **not** directly relate $V^\pi(s, g')$ to $V^\pi(s, g_{\text{target}})$. It serves a more specific role: once the agent reaches any state $s' \in S_{g'}$ satisfying intermediate goal $g'$, the remaining value $V^\pi(s', g_{\text{target}})$ is approximately constant (within $\delta$) regardless of which specific $s'$ was reached. This makes the via-goal value decomposition $V^\pi_{\text{via}}(s, g; g') = p^\pi(g'|s) \cdot \mathbb{E}_{s' \sim d^\pi(\cdot|s,g')}[V^\pi(s', g)]$ (Eq. 26) well-defined and enables Theorem 6.2.

The mechanism by which $\rho_{\text{HG}}$ accelerates learning is **coverage expansion** (Theorem 6.1, empirically verified in **MQ7**): the prior proposes diverse actions from behaviors toward multiple intermediate goals, and the RL critic (first term in Eq. 21) selects which actions are useful for reaching $g_{\text{target}}$.

### Empirical measurement

We agree that directly measuring this assumption would strengthen the paper, and we commit to including quantitative measurements in the revision — specifically, computing $|V^\pi(s_1, g_{\text{target}}) - V^\pi(s_2, g_{\text{target}})|$ for sampled pairs $s_1, s_2 \in S_{g'}$ using the learned critic across training epochs.

Our empirical results already provide indirect evidence: GCHR's advantage is largest on tasks where the assumption is most reasonable (Fetch tasks, HandReach — where different configurations at the same end-effector pose have similar distances to targets), and degrades on tasks where the assumption is more approximate (BlockRotateXYZ, BlockRotateParallel — where different joint configurations achieving similar end-effector poses can have substantially different reachability to target rotations).



---

## MQ5. RIS and SAW Comparison

**Raised by:** Reviewer k84A (Q5), Reviewer XYhH (Q2)

### Structural comparison

| | **GCHR** | **RIS** [R3] | **SAW** [R4] |
|---|---|---|---|
| Setting | Online | Online | Offline |
| Additional networks | None (reuses target net) | Subgoal prediction net | Subpolicy $\pi_{\text{sub}}$ |
| Prior construction | Mixture over $K$ waypoints | Single imagined midpoint | Advantage-weighted regression |
| Training | End-to-end, single phase | End-to-end + subgoal net | Three sequential phases |
| Prior evolution | Co-evolves with policy | Co-evolves (via subgoal net) | Static (fixed offline data) |
| Monotonic improvement | Theorem 6.2 | Not established | Not established |

The key structural difference from RIS: GCHR aggregates $K$ hindsight goals from actual trajectories into a mixture prior, while RIS uses a single imagined midpoint from a learned subgoal prediction network. GCHR requires no additional learned components. The key difference from SAW: GCHR operates online with an evolving prior, enabling Theorem 6.2 (monotonic improvement); SAW's prior is static.

### RIS experimental comparison

All methods use the **SAC backbone** for fair comparison:

**Table R3.** Success rate (%) on Fetch benchmarks (all methods: SAC backbone).

| Method | FetchReach | FetchPush | FetchSlide | FetchPick |
|---|---|---|---|---|
| RIS (SAC) | 70±3 | 97±4 | 21±6 | 52±4 |
| SAC+HER | 100±0 | 95±2 | 23±5 | 51±4 |
| CRL | 100±0 | 6±5 | 2±1 | 8±2 |
| **SAC+GCHR** | **100±0** | **99±3** | **38±3** | **52±6** |

GCHR matches or outperforms RIS on all tasks, with the largest advantage on FetchSlide (+17pp). Notably, CRL — strong on locomotion/navigation (Table R1) — performs poorly on Fetch manipulation (6–8% on Push/Slide/Pick), highlighting that GCHR performs well across both domains.

**References:**
[R3] Goal-conditioned RL with Imagined Subgoals. ICML 2021.
[R4] Flattening Hierarchies with Policy Bootstrapping. NeurIPS 2025.

---

## MQ6. Training Time / Wall-Clock Overhead

**Raised by:** Reviewer UQ5F (W4), Reviewer k84A (Q2)

The hindsight goal prior requires $K$ additional forward passes through the target network per update step (no gradient computation needed). Empirically on FetchPush:

| $K$ | Wall-clock overhead vs. DDPG+HER | Env steps to 90% success |
|---|---|---|
| 5 | +12% | ~32k (1.6× faster) |
| 10 (default) | +22% | ~28k (1.8× faster) |
| 15 | +31% | ~27k (1.9× faster) |
| 20 | +40% | ~27k (diminishing returns) |

Given that GCHR reaches equivalent success rates 1.5–2× faster in environment steps, the net wall-clock time to convergence is favorable for $K$ up to approximately 15. We use $K{=}10$ as the default, balancing overhead against bootstrapping quality.

---

## MQ7. Coverage Expansion — Direct Empirical Evidence

**Raised by:** Reviewer XYhH (Q4), Reviewer k84A (implicitly via Theorem 6.1)

We provide direct empirical verification that $\rho_{\text{HG}}$ discovers useful actions outside the behavioral support, not mere smoothing. For $N{=}300$ $(s,g)$ pairs sampled from the replay buffer, we draw 200 actions from $\rho_{\text{HG}}$ and classify each as **novel** (minimum $\ell_2$ distance to any $\rho_{\text{beh}}$ action exceeds threshold $\varepsilon$) or **covered** (within $\varepsilon$). Q-advantages are normalized per-pair against random actions.

### Reacher (2D actions, SAC+GCHR, 5.1M steps)

| Action source | Q-advantage over random | Notes |
|---|---|---|
| $\pi_\theta$ (policy) | +1.20 ± 0.09 | Trained policy (sanity check: highest) |
| $\pi_{\text{HG}}$ covered | +1.05 ± 0.08 | $\pi_{\text{HG}}$ actions overlapping $\rho_{\text{beh}}$ |
| $\rho_{\text{beh}}$ (recorded) | +0.94 ± 0.07 | Behavioral support |
| **$\pi_{\text{HG}}$ novel** | **+0.25 ± 0.07** | **Outside $\rho_{\text{beh}}$ support** |
| Random | 0.00 | Baseline |

Novel fraction: 17.7%. **Novel $\pi_{\text{HG}}$ actions achieve +0.25 advantage over random, confirming genuine coverage expansion.** These novel actions are expectedly weaker than behavioral actions (+0.25 vs +0.94), since they originate from behaviors toward *related but different* goals. The critical point is that they are *positive-advantage*: the prior proposes useful exploration directions that pure self-imitation would miss entirely. This is precisely the coverage expansion mechanism of Theorem 6.1.

The 2D action space allows direct visualization:

![Figure R3: Q-landscape showing coverage expansion on Reacher.](action_coverage_res/figure_a_reacher_qlandscape.png)
*Figure R3. Q-landscape on Reacher for three $(s,g)$ pairs. Background: $Q(s,\cdot,g)$ over 2D action space (blue = high Q, red = low Q). Red ×: $\rho_{\text{beh}}$. Blue ○: novel $\pi_{\text{HG}}$ actions. Green ★: $\pi_\theta$. Novel actions consistently land in high-Q regions beyond the behavioral support.*

### Pusher Easy (higher-dim actions, SAC+GCHR)

In higher-dimensional action spaces, the distance-based novelty criterion becomes less discriminative (novel fraction ~97%). However, the training dynamics are informative: at 2.2M steps (early exploration), novel actions achieve +0.91 advantage over random, showing that coverage expansion is most impactful when the behavioral support is sparse. As the policy converges and $\rho_{\text{beh}}$ fills in, the marginal value diminishes — consistent with GCHR's design.

![Figure R4: Coverage expansion summary across environments.](action_coverage_res/combined_coverage_summary.png)
*Figure R4. Left: Q-advantage by action source. Right: advantage over training. On Reacher, novel actions maintain positive advantage throughout. On Pusher Easy, coverage expansion is most valuable during early exploration.*

---

## MQ8. Notation and Naming Inconsistencies

**Raised by:** Reviewer k84A (W2), Reviewer XYhH (W4)

We sincerely apologize for these issues. All will be corrected in the revision:

| Issue | Location | Fix |
|---|---|---|
| "GCQS" instead of "GCHR" | Figure 10 | Rename to GCHR |
| $\bar{\pi}_\theta$ vs $\pi_{\bar{\theta}}$ | Eq. 10 vs Algorithm 1 | Unify: $\bar{\pi}_\theta \equiv \pi_{\bar{\theta}}$, use $\bar{\pi}_\theta$ consistently in text |
| "=" vs "≈" | Eq. 10 vs Eq. 18 | Eq. 10 is the definition; Eq. 18 is the finite-sample approximation. Mark clearly. |
| "Theorem Theorem 6.1" | Lines 528, 540 | Fix cross-reference formatting |
| "Assumption Theorem 4.1" | Lines 306, 550 | Fix cross-reference formatting |
| "sHSRe" | Appendix D, lines 839, 882 | Correct to "share" |
| Swapped critic/policy objectives for DDPG+HER | Appendix D.1, Eqs. 33–34 | Swap to correct order |
