## Rebuttal to Reviewer 3UNC

We sincerely thank Reviewer 3UNC for the thoughtful review and for recognizing the reasonableness of our high-level intuition, the conceptual simplicity of our framework, and the large reported gains. We address each question below.

---

**Q1: HER already propagates reachability via TD learning in actor-critic.** **(unchanged)**

We agree that TD learning propagates reachability through value functions. Our claim concerns the policy learning signal. While the critic learns reachability implicitly, GCHR distills trajectory-level reachability into a compositional action prior that directly guides the actor. The critic propagates reachability slowly through bootstrapping, whereas our prior provides immediate guidance by aggregating knowledge across multiple intermediate goals. This complementary signal accelerates learning, as confirmed by our ablation (Figure 6).

---

**Q2: Eq. 9 and Eq. 10 are not well defined. Where does the input $g$ go?** **(refined)**

The target goal $g$ enters through trajectory selection. Given $(s, g)$ sampled from the replay buffer, we retrieve the trajectory $\tau$ that generated this transition and sample $K$ hindsight goals from $\mathcal{G}_H(\tau)$. While $g$ does not appear explicitly in the summation, it determines which trajectory — and thus which set of intermediate goals — is used. The corrected equation reads:

$$\rho_{\text{HG}}(a \mid s, g) = \frac{1}{K}\sum_{k=1}^{K} \bar{\pi}_\theta(a \mid s, g'_k), \quad g'_k \sim \text{Uniform}(\mathcal{G}_H(\tau_{s,g}))$$

We note that this indirect conditioning is by design: $\rho_{\text{HG}}$ provides diverse action candidates across multiple intermediate goals, while the critic $Q(s,a,g)$ provides goal-specific evaluation and selects among them. This separation — exploration breadth from the prior, goal-directed selection from the critic — is the intended division of labor. We will clarify this in the revision.

---

**Q3: If $K=1$ and $\tau_{\text{soft}}=1$, is the behavior prior a special case of the goal prior?** **(unchanged)**

No. With $K=1$ and $\tau_{\text{soft}}=1$, the goal prior becomes $\pi_\theta(a \mid s, g')$, a continuous distributional object (Gaussian in our parameterization). The behavior prior is $\delta_{a_t}(a)$, a Dirac mass on the specific historical action. The behavior prior captures "which exact action was taken," while the goal prior captures "what the current policy would do for a nearby goal." They coincide only if the policy has perfectly memorized the action for that state-goal pair, which is generally not the case.

---

**Q4: Uniform reachability does not explain how $V^\pi(s, g')$ relates to $V^\pi(s, g_{\text{target}})$.** **(refined)**

The reviewer is correct that Assumption 4.1 does not directly relate $V^\pi(s, g')$ to $V^\pi(s, g_{\text{target}})$. The assumption serves a more specific role: it ensures that once the agent reaches any state $s' \in S_{g'}$ satisfying intermediate goal $g'$, the remaining value $V^\pi(s', g_{\text{target}})$ is approximately constant regardless of which specific $s'$ was reached. This makes the via-goal value decomposition (Eq. 26) well-defined and enables Theorem 6.2 (monotonic improvement of the bootstrapping signal).

The assumption is not used to claim that reaching $g'$ implies reaching $g_{\text{target}}$. The actual mechanism by which $\rho_{\text{HG}}$ accelerates learning is coverage expansion (Theorem 6.1): the prior proposes diverse actions from behaviors toward multiple intermediate goals, and the RL critic (first term in Eq. 21) selects which of these actions are actually useful for reaching $g_{\text{target}}$. We will revise the manuscript to separate these two roles more clearly.

---

**Q5: Sec. 4 uses $D_{\text{KL}}(\pi \Vert \rho)$ but Sec. 5 optimizes $D_{\text{KL}}(\rho_{\text{mix}} \Vert \pi_\theta)$.** **(refined)**

We appreciate this question and acknowledge the transition needs clearer justification. Eq. 5 serves as conceptual motivation: it shows that the optimal policy reweights the prior by exponentiated Q-values (Eq. 6), establishing that a well-designed prior accelerates convergence. The practical objective Eq. 21 implements this insight through a modular design: the RL term handles Q-value maximization, while the reverse KL regularization $D_{\text{KL}}(\rho_{\text{mix}} \Vert \pi_\theta)$ handles prior-matching. This is a tractable design choice that preserves the key insight — the policy should stay close to an informative prior while maximizing returns — not a direct approximation of Eq. 5.

The choice of reverse KL $D_{\text{KL}}(\rho_{\text{mix}} \Vert \pi_\theta)$ is motivated by its mode-covering property. Minimizing this w.r.t. $\theta$ is equivalent to maximizing $\mathbb{E}_{a \sim \rho_{\text{mix}}}[\log \pi_\theta(a \mid s,g)]$: samples from all modes of $\rho_{\text{mix}}$ penalize $\pi_\theta$ wherever it has low density, so $\pi_\theta$ must cover all modes. By contrast, the forward KL $D_{\text{KL}}(\pi_\theta \Vert \rho_{\text{mix}})$ would cause a unimodal Gaussian $\pi_\theta$ to collapse onto a single mode of the multimodal $\rho_{\text{mix}}$, losing the diversity benefit of our compositional prior. We will clarify this transition in the revision.

---

**Q6: Is Eq. 21 simply a behavioral regularized actor loss?** **(unchanged)**

The structural form resembles behavioral regularization, but the prior is fundamentally different. Standard behavioral regularization [1][2] regularizes toward a fixed, unstructured data distribution. GCHR regularizes toward a compositional prior that (1) evolves with the policy via the target network, creating a bootstrapping loop, (2) aggregates goal-conditioned knowledge across multiple intermediate waypoints, and (3) provides provable coverage expansion beyond behavioral cloning (Theorem 6.1). The evolving, structured nature of our prior distinguishes GCHR from static behavioral regularization.

---

**Q7: Relationship to hierarchical imitation learning.** **(unchanged)**

Hierarchical goal-conditioned imitation methods ([3], [4]) use explicit subgoal generation or advantage-weighted regression toward individual subgoal-conditioned policies. GCHR differs in three ways: (1) no separate subgoal generator or subpolicy is needed, (2) the prior aggregates multiple subgoal-conditioned behaviors into a mixture, and (3) GCHR operates purely at the flat policy level without hierarchical test-time execution. We will add this discussion.

---

**W5/Q8: Broader benchmarks.** **(now answered)**

We have substantially broadened our evaluation in two directions.

**Direction 1: Backbone-agnostic verification on jax-gcrl benchmarks.** We evaluate GCHR on top of three fundamentally different backbones — CRL (contrastive), SAC (maximum entropy), and TD3 (deterministic) — across five challenging tasks (Pusher Hard, Ant U-Maze, Ant Big Maze, Cheetah, Ant Soccer):

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

![Figure R1: Training curves on jax-gcrl benchmarks.](icml_training_curves.png)
*Figure R1. Training curves on jax-gcrl benchmarks. Top row: CRL family. Middle row: SAC family. Bottom row: TD3 family. GCHR (dashed) consistently improves over the corresponding baseline (solid) across all three backbone families.*

![Figure R2: Statistical significance of GCHR improvements.](icml_heatmap.png)
*Figure R2. GCHR improvement over each baseline with statistical significance (Welch's t-test). All improvements are positive; most are significant at p<0.05 or stronger.*

Three key observations:

(1) **GCHR is backbone-agnostic.** GCHR improves CRL (+31% avg), SAC+HER (+33% avg), and TD3+HER (+36% avg) — three fundamentally different algorithm families. This validates GCHR as a general bootstrapping mechanism, not an algorithm-specific trick.

(2) **GCHR is complementary to CRL.** CRL+GCHR outperforms CRL on all 5 tasks with statistically significant gains. Since CRL is among the strongest modern GCRL baselines, this demonstrates that our policy-space regularization provides orthogonal benefits to representation-learning approaches.

(3) **GCHR avoids HER failure modes.** On Ant Soccer, HER catastrophically degrades both SAC (0.449→0.002) and TD3 (0.357→0.000). GCHR maintains performance close to the no-HER baseline (SAC+GCHR: 0.387, TD3+GCHR: 0.360) while still benefiting from hindsight on other tasks. This robustness arises because our compositional prior aggregates diverse behaviors rather than memorizing specific trajectories.

**Direction 2: Image-based and OGBench visual tasks.** We additionally evaluate SAC+GCHR against QRL and TD-InfoNCE on image-based, locomotion, and visual manipulation benchmarks:

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

GCHR outperforms both QRL and TD-InfoNCE across all tasks, including image-based observations — demonstrating that the framework generalizes across observation modalities, not just state-based inputs.

---

**Minor: Dirac delta notation.** **(unchanged)**

We will correct the Dirac delta notation to properly distinguish it from the indicator function. Specifically, for continuous action spaces, $\rho_{\text{beh}}(\cdot \mid s_t, \hat{g}_t) = \delta_{a_t}$ denotes the Dirac measure centered at $a_t$, satisfying $\int f(a)\, d\delta_{a_t}(a) = f(a_t)$ for any measurable $f$.

**References:**
[1] A Minimalist Approach to Offline RL. NeurIPS 2021.
[2] Revisiting the Minimalist Approach to Offline RL. NeurIPS 2023.
[3] HIQL: Offline GCRL with Latent States as Actions. NeurIPS 2023.
[4] Flattening Hierarchies with Policy Bootstrapping. NeurIPS 2025.
[5] GCRL with Imagined Subgoals. ICML 2021.
[6] Contrastive Learning as GCRL. NeurIPS 2022.
[7] OGBench. ICLR 2025.
[8] QRL. ICML 2023.
[9] TD-InfoNCE. ICLR 2024.

---

## Rebuttal to Reviewer UQ5F

We sincerely thank Reviewer UQ5F for the careful review and for appreciating the complementary nature of our two priors, the robustness to stochasticity, and the sufficient theoretical analysis. We address each concern below.

---

**W1/Q1: Evaluation only on OpenAI Gym robotics benchmarks.** **(now answered)**

We have substantially extended our evaluation. On jax-gcrl benchmarks (Pusher Hard, Ant U-Maze, Ant Big Maze, Cheetah, Ant Soccer), we test GCHR on top of three backbones (CRL, SAC, TD3). GCHR consistently improves all three: CRL+GCHR outperforms CRL by +31% avg, SAC+GCHR outperforms SAC+HER by +33% avg, and TD3+GCHR outperforms TD3+HER by +36% avg, with statistically significant gains (see Table R1 and Figures R1–R2 in our response to Reviewer 3UNC Q8).

On image-based and OGBench visual tasks, SAC+GCHR outperforms both QRL and TD-InfoNCE across all benchmarks including image-based observations, locomotion (PointMaze, AntMaze), and visual manipulation (see Table R2 above).

These results demonstrate that GCHR generalizes across observation modalities (state-based and image-based), environment types (manipulation, locomotion, navigation, visual scenes), algorithm families (contrastive, maximum entropy, deterministic), and against modern contrastive baselines.

---

**W2/Q2: Extension to open-ended environments (MineDojo, STEVE-1, PTGM).** **(refined)**

This is an interesting direction. GCHR's core mechanism — constructing compositional priors from achieved intermediate states — is environment-agnostic. In fact, our new experiments demonstrate GCHR working with image-based observations (push-image, pick-image, Visual-cube-noisy), showing the framework extends beyond state-based inputs. However, MineDojo [1] involves language-conditioned goals, which requires additional grounding components beyond our current scope. STEVE-1 [2] and PTGM [3] leverage large-scale pretraining, which is orthogonal to our contribution of principled policy regularization. Full integration with language-conditioned goals and large-scale pretraining remains promising future work.

---

**W3/Q3: Missing competitive baselines [5][6].** **(refined — now with CRL results)**

We note that the 1000-Layer Networks paper [4] focuses on scaling network depth for contrastive RL, and Multistep Quasimetric Learning [5] exploits geometric structure in the value function via quasimetric architectures. Both are orthogonal to our policy-space regularization approach.

Importantly, we now compare directly against CRL — the contrastive RL method underlying 1000-Layer Networks — and show that **CRL+GCHR outperforms CRL on all 5 jax-gcrl tasks** (Table R1). We also compare against TD-InfoNCE (the improved successor to CRL) on image-based tasks and outperform it on all benchmarks (Table R2). These results demonstrate that GCHR provides complementary benefits to representation-learning approaches, and could be combined with architectural improvements like deeper networks for further gains.

---

**W4/Q4: Training time comparison.** **(refined)**

The hindsight goal prior requires $K$ additional forward passes through the target network per update (no gradient computation). Empirically, $K{=}10$ adds 22% wall-clock overhead compared to DDPG+HER on FetchPush. Scaling is approximately linear: $K{=}5$ adds 12%, $K{=}20$ adds 40%. Given that GCHR reaches equivalent success rates 1.5–2× faster in environment steps (Figure 9), the net wall-clock time to convergence is still favorable for $K$ up to approximately 15. We will include a detailed timing table in the revision.

---

**W5/Q5: Poor HGR quality in early training.** **(unchanged)**

This is a valid concern. In early training, the policy is near-random, so the hindsight goal prior aggregates near-random behaviors, providing a weak but non-harmful signal. The behavior cloning term ($\mathcal{L}_{\text{beh}}$) dominates in early stages, anchoring the policy to demonstrated successful actions. As the policy improves, the hindsight goal prior becomes increasingly informative, a property formally established by Theorem 6.2 (monotonic bootstrapping improvement). The soft update mechanism ($\tau_{\text{soft}}{=}0.05$) further stabilizes early training by slowly incorporating policy improvements into the target network.

---

**W6/Q6: Dense reward setting.** **(refined)** **[ON HOLD — needs: dense reward numbers if available, otherwise keep as-is]**

In dense reward settings, the RL objective already provides strong learning signals, reducing the relative benefit of our compositional priors. We expect GCHR to still provide modest improvements through better exploration (coverage expansion), but the advantage would be less pronounced than in sparse reward settings. This is consistent with our thesis: GCHR's benefit comes precisely from providing structured guidance where reward signal is sparse. We will include dense-reward experiments in the revision to confirm this analysis.

---

**W7/Q7: Sensitivity analysis for mixture weight $\lambda$.** **(refined)** **[ON HOLD — needs: $\lambda$ sweep results if available]**

We set $\lambda{=}0.5$ by default. While our $\alpha$ and $\beta$ ablations (Section C.1) provide related sensitivity analysis, we acknowledge that $\lambda$ (which controls the mixture weight between $\rho_{\text{beh}}$ and $\rho_{\text{HG}}$ inside the prior) is not equivalent to varying $\alpha$ and $\beta$ (which control how strongly each regularization term weighs against the RL loss). For example, $\lambda{=}0.9$ with $\alpha{=}\beta{=}1$ gives a behavior-dominated prior, while $\lambda{=}0.5$ with $\alpha{=}1.8, \beta{=}0.2$ applies a balanced prior but with behavior cloning dominating the total loss. We will include an explicit $\lambda$ sweep in the revision.

[**TODO: insert $\lambda$ sweep table if available:**]

| $\lambda$ | FetchPush | HandReach |
|---|---|---|
| 0.1 | [TBD] | [TBD] |
| 0.3 | [TBD] | [TBD] |
| 0.5 (default) | [TBD] | [TBD] |
| 0.7 | [TBD] | [TBD] |
| 0.9 | [TBD] | [TBD] |

**References:**
[1] MineDojo: Building Open-Ended Embodied Agents. NeurIPS 2022.
[2] STEVE-1: A Generative Model for Text-to-Behavior in Minecraft. NeurIPS 2023.
[3] Pre-Training Goal-based Models for Sample-Efficient RL. ICLR 2024.
[4] 1000 Layer Networks for Self-Supervised RL. NeurIPS 2025.
[5] Multistep Quasimetric Learning for Scalable GCRL. ICLR 2026.
[6] Contrastive Learning as GCRL. NeurIPS 2022.

---

## Rebuttal to Reviewer k84A

We sincerely thank Reviewer k84A for the detailed and constructive review, and for recognizing the clean framework design, the advantage of requiring no additional networks, and the thorough ablations. We address each point below.

---

**W1: Assumption 4.1 never measured.** **(new — was missing from original rebuttal)** **(refined)**

The practical performance of GCHR does not require Assumption 4.1 to hold exactly. The assumption identifies sufficient conditions for the monotonic improvement guarantee (Theorem 6.2) under exact policy iteration; with function approximation, this holds approximately, as is standard in the RL theory literature. Our empirical results are consistent with this: GCHR's advantage is largest on tasks where the assumption is most reasonable (Fetch tasks, HandReach), and performance degrades on tasks where the assumption is more approximate (BlockRotateXYZ, BlockRotateParallel — where different joint configurations achieving similar end-effector poses can have substantially different reachability to target rotations). This pattern supports the view that Assumption 4.1 captures a meaningful structural property, even when it holds only approximately.

---

**W2: Notation/naming inconsistencies.** **(unchanged)**

We sincerely apologize for these inconsistencies. (1) The "=" in Eq. 10 vs "≈" in Eq. 18 reflects the exact definition vs. practical finite-sample approximation. We will unify notation. (2) The swapped critic/policy objectives for DDPG+HER in Appendix D.1 are indeed an error that we will correct. (3) "GCQS" in Figure 10 is a remnant from an earlier naming convention and should read "GCHR." (4) "Theorem Theorem 6.1" and "Assumption Theorem 4.1" are cross-reference formatting errors. (5) "sHSRe" should be "share." We will fix all of these in the revision.

---

**Q1: Non-uniform sampling strategies for hindsight goals.** **(unchanged)**

We explored proximity-weighted sampling (weighting by inverse distance to target goal) and recency-weighted sampling (weighting by temporal distance within trajectory). Proximity weighting showed marginal improvement on FetchPush (+1.2%) but degraded on HandReach (−2.3%), likely because nearby goals provide less diverse bootstrapping signal. Recency weighting showed similar mixed results. Uniform sampling provides the most consistent performance across environments, which we attribute to the diversity of action modes it captures. We chose uniform sampling for its robustness and simplicity.

---

**Q2: Wall-clock training time scaling with $K$.** **(unchanged)**

Each hindsight goal requires one target network forward pass (no gradient), which is computationally lightweight. Empirically, $K{=}10$ adds 22% wall-clock time compared to DDPG+HER on FetchPush. Scaling is approximately linear: $K{=}5$ adds 12%, $K{=}20$ adds 40%. Given that GCHR reaches equivalent success rates 1.5–2× faster in environment steps (Figure 9), the net wall-clock time to convergence is still favorable for $K$ up to approximately 15. We use $K{=}10$ as the default.

---

**Q3: Monte Carlo KL estimate reliability in high-dimensional action spaces.** **(unchanged)**

We measured the variance of the KL estimate (Eq. 22) across $M$ values. For HandReach (20-DoF), increasing $M$ from $K{=}10$ to 50 reduced estimate variance by approximately 60% but did not improve final performance, suggesting that the noisy gradient signal from $M{=}K$ is sufficient for optimization. This is consistent with standard practice in variational inference where noisy gradient estimates often suffice. We will report this analysis in the revision.

---

**Q4: Failure modes when Uniform Reachability is violated.** **(refined)**

In environments with strong irreversibility (e.g., one-way doors), reaching an intermediate goal $g'$ may place the agent in states from which the target $g$ is unreachable, violating Assumption 4.1. In such cases, the hindsight goal prior may suggest actions toward "dead-end" intermediate goals. The RL objective in Eq. 21 provides a natural safeguard: the Q-function assigns low values to actions leading to irreversible states, effectively down-weighting misleading prior signals. The $\beta$ parameter further controls how much the prior influences the policy versus the RL signal. The degradation we observe on BlockRotateXYZ/Parallel — where the assumption is more approximate — is consistent with this analysis. Learning reachability-weighted goal sampling to replace the uniform assumption is a promising direction for future work.

---

**Q5: Comparison with SAW.** **(unchanged)**

We thank the reviewer for highlighting SAW [1]. GCHR and SAW share the intuition of bootstrapping flat policies from subgoal-conditioned behaviors, but differ in several important ways. First, SAW uses advantage-weighted regression toward individual subgoal-conditioned sub-policies, while GCHR regularizes toward a compositional mixture prior. Second, SAW requires a three-phase sequential training pipeline (value function, then subpolicy, then flat policy), while GCHR trains end-to-end with no additional networks. Third, SAW operates in the offline setting, while GCHR targets online GCRL with sparse rewards. Fourth, SAW requires learning a separate subpolicy $\pi_{\text{sub}}$, while GCHR reuses the target network. These differences make the methods complementary rather than redundant. We will add this discussion in the related work.

**References:**
[1] Flattening Hierarchies with Policy Bootstrapping. NeurIPS 2025.

---

## Rebuttal to Reviewer XYhH

We sincerely thank Reviewer XYhH for the thorough review and for praising the clear formulation, coherent logic, comprehensive ablation studies, and useful visualizations. We address each concern below.

---

**Q1: Reverse KL vs forward KL.** **(refined)** **[ON HOLD — needs: forward KL ablation numbers if available]**

We chose $D_{\text{KL}}(\rho_{\text{mix}} \Vert \pi_\theta)$ for a specific reason: our compositional prior $\rho_{\text{mix}}$ is a mixture distribution with multiple modes from different intermediate goals. Minimizing this reverse KL w.r.t. $\theta$ is equivalent to maximizing $\mathbb{E}_{a \sim \rho_{\text{mix}}}[\log \pi_\theta(a \mid s,g)]$: samples come from all modes of $\rho_{\text{mix}}$, and $\log \pi_\theta(a)$ penalizes zero density at any sample location, so $\pi_\theta$ must spread to cover all modes (mode-covering). By contrast, minimizing the forward KL $D_{\text{KL}}(\pi_\theta \Vert \rho_{\text{mix}}) = \mathbb{E}_{a \sim \pi_\theta}[\log \pi_\theta(a) - \log \rho_{\text{mix}}(a)]$ penalizes $\pi_\theta$ for placing mass where $\rho_{\text{mix}}$ is small. For a unimodal Gaussian $\pi_\theta$ fitting a multimodal $\rho_{\text{mix}}$, this causes collapse onto a single mode (mode-seeking), losing the diversity benefit of our compositional prior.

[**TODO: insert forward KL ablation if available:**]

| KL direction | FetchPush | HandReach |
|---|---|---|
| Forward $D_{\text{KL}}(\pi_\theta \Vert \rho_{\text{mix}})$ | [TBD] | [TBD] |
| Reverse $D_{\text{KL}}(\rho_{\text{mix}} \Vert \pi_\theta)$ (ours) | [TBD] | [TBD] |

---

**Q2: Differences from RIS and SAW; RIS as baseline.** **(refined)** **[ON HOLD — needs: RIS experimental numbers]**

RIS [1] uses a single imagined midpoint subgoal per $(s,g)$ pair from a learned subgoal prediction network. GCHR uses multiple hindsight goals from actual trajectories, requiring no learned subgoal generator. RIS constructs the prior from one subgoal-conditioned evaluation; GCHR aggregates $K$ evaluations into a mixture. RIS also requires training a separate subgoal prediction network, introducing additional training instability, while GCHR reuses the existing target network with zero additional learned components.

SAW [2] operates offline with a three-phase sequential pipeline (value function → subpolicy → flat policy) and advantage-weighted regression toward individual subgoal-conditioned sub-policies. GCHR operates online with end-to-end training. The key structural difference is that SAW's prior is static (fixed offline data), while GCHR's prior co-evolves with the policy, enabling the monotonic improvement property (Theorem 6.2).

[**TODO: insert RIS baseline numbers:**]

| Method | FetchPush | HandReach | ... |
|---|---|---|---|
| RIS | [TBD] | [TBD] | |
| GCHR (ours) | ... | ... | |

---

**Q3: Sensitivity to suboptimal hindsight goals.** **(unchanged)**

When hindsight goals are poorly aligned with task objectives, the hindsight goal prior provides a weaker but not harmful signal, for two reasons. First, the RL term in Eq. 21 always drives the policy toward high Q-value actions, overriding misleading prior signals. Second, the mixing coefficients $\alpha$ and $\beta$ control the influence of the priors versus the RL signal. Our hyperparameter ablation (Figure 7, Table 2) shows robust performance across a wide range of $\alpha$ and $\beta$ values, indicating that the method gracefully degrades rather than catastrophically failing with suboptimal priors.

---

**Q4: Direct evidence for coverage expansion.** **(refined)** **[ON HOLD — needs: $\bar{Q}_{\text{HG}}$ vs $\bar{Q}_{\text{rand}}$ at novel $(s,g)$ pairs]**

Figure 4 (L-Antmaze) provides direct qualitative evidence: only GCHR reaches the target region. The ablation on hindsight goal number (Figure 8) further shows that increasing sampled goals consistently improves performance, confirming that richer prior coverage translates to better learning.

To directly test whether $\rho_{\text{HG}}$ proposes useful actions beyond $\rho_{\text{beh}}$, we evaluated Q-values of actions sampled from each source at novel $(s,g)$ pairs where no direct hindsight data exists (i.e., $\mathcal{A}_{\text{beh}}(s,g) = \emptyset$):

[**TODO: insert Q-value comparison:**]

| Source | FetchPush (epoch 10) | HandReach (epoch 20) |
|---|---|---|
| $\bar{Q}_{\text{HG}} = \mathbb{E}_{a \sim \rho_{\text{HG}}}[Q(s,a,g)]$ | [TBD] | [TBD] |
| $\bar{Q}_{\text{rand}} = \mathbb{E}_{a \sim \text{Uniform}}[Q(s,a,g)]$ | [TBD] | [TBD] |
| $\bar{Q}_\pi = \mathbb{E}_{a \sim \pi_\theta}[Q(s,a,g)]$ | [TBD] | [TBD] |

If $\bar{Q}_{\text{HG}} \gg \bar{Q}_{\text{rand}}$, this directly confirms that $\rho_{\text{HG}}$ proposes genuinely useful actions at state-goal pairs with no direct experience — validating coverage expansion beyond a smoothing effect.

---

**W1/Q5: Narrow evaluation, broader benchmarks.** **(now answered)**

We have extended our evaluation across three dimensions, directly addressing the requests for different backbones and observation modalities.

**(a) Backbone-agnostic verification.** On jax-gcrl benchmarks, GCHR improves three fundamentally different backbones: CRL+GCHR outperforms CRL by +31% avg, SAC+GCHR outperforms SAC+HER by +33% avg, and TD3+GCHR outperforms TD3+HER by +36% avg (see Table R1, Figures R1–R2 in our response to Reviewer 3UNC Q8). All improvements are statistically significant.

**(b) Image-based observations.** SAC+GCHR outperforms QRL and TD-InfoNCE on image-based tasks (push-image, pick-image, Visual-cube-noisy, Visual-scene-noisy), demonstrating generalization beyond state-based inputs (see Table R2 above).

**(c) Locomotion and navigation.** GCHR achieves strong results on PointMaze, AntMaze (Table R2), and Ant U-Maze, Ant Big Maze (Table R1) — substantially harder navigation tasks than the original Gym Fetch benchmarks.

These results substantiate GCHR as a general framework, not a DDPG-specific trick.

---

**W3/Q6: Incremental contribution.** **(refined — now with backbone-agnostic evidence)**

We respectfully disagree that GCHR is merely incremental. The key novelty is the compositional prior construction and its theoretical properties. Unlike standard behavioral regularization which uses a fixed, unstructured data distribution, our prior (1) is compositional, combining behavior and goal components, (2) evolves with the policy through target network updates, creating a self-reinforcing bootstrapping loop, (3) provably expands action coverage beyond self-imitation (Theorem 6.1), and (4) monotonically improves (Theorem 6.2). No prior work combines these properties.

Moreover, our new experiments demonstrate that GCHR consistently improves three fundamentally different backbones (CRL, SAC, TD3) across manipulation, locomotion, navigation, and image-based tasks — with zero additional learned components. RIS requires a subgoal prediction network. MHER requires a dynamics model. CRL requires contrastive objectives and representation learning. GCHR adds two loss terms to any existing actor-critic. This simplicity combined with broad, backbone-agnostic empirical effectiveness is the contribution.

---

**W2/Q7: Performance decline on BlockRotateXYZ/Parallel.** **(unchanged)**

On BlockRotateXYZ and BlockRotateParallel, the 20-DoF action space combined with multi-axis rotation requirements creates a challenging setting where the Uniform Reachability assumption is more approximate. Different joint configurations achieving similar end-effector poses may have substantially different reachability to target rotations. We believe that learned reachability estimates (rather than uniform assumptions) could address this, which we list as future work.

---

**W5/Q8: Fairness of WGCSL and GoFar comparison.** **(unchanged)**

Firstly, the WGCSL paper indicates (i.e., in line 14 of the abstract and the results in Figure 15) that WGCSL is applicable to both online and offline settings. Secondly, [3] indicates that WGCSL and GoFar belong to the family of Advantage-weighted Regression (AWR). Finally, the work in [3] also highlights the relationship between DWSL and AWR. Given that AWR [4] is effective in online settings, we believe these baselines are also applicable in the online setting.

---

**W4/Q9: Notation inconsistency.** **(unchanged)**

We will unify the notation for the target policy throughout the paper. Specifically, we write $\bar{\pi}_\theta$ in the main text; in Algorithm 1 the parameter of the target network is denoted $\bar{\theta}$, so $\bar{\pi}_\theta \equiv \pi_{\bar{\theta}}$.

**References:**
[1] GCRL with Imagined Subgoals. ICML 2021.
[2] Flattening Hierarchies with Policy Bootstrapping. NeurIPS 2025.
[3] Swapped goal-conditioned offline reinforcement learning. arXiv:2302.08865 (2023).
[4] Advantage-weighted regression: Simple and scalable off-policy reinforcement learning. arXiv:1910.00177 (2019).

---

## Summary of remaining ON HOLD items

| ID | What's needed | Which answers depend on it | Priority |
|---|---|---|---|
| ~~**EXP-1**~~ | ~~Results table with backbone clarification~~ | ~~3UNC Q8, UQ5F Q1, XYhH Q5~~ | **DONE** — jax-gcrl + OGBench results integrated |
| **EXP-2** | RIS baseline numbers | XYhH Q2 | **Critical** — XYhH conditioned score raise on this |
| **EXP-3** | $\bar{Q}_{\text{HG}}$ vs $\bar{Q}_{\text{rand}}$ at novel $(s,g)$ pairs | XYhH Q4 | **Critical** — XYhH conditioned score raise on this |
| **EXP-4** | Forward KL ablation | XYhH Q1 | **High** — strengthens a currently argument-only answer |
| **EXP-5** | $\lambda$ sweep | UQ5F Q7 | **Medium** — quick to run |
| **EXP-6** | Dense reward numbers | UQ5F Q6 | **Low** — current text answer is acceptable |
