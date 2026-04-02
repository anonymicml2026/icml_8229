
# Hindsight Goal Corruption — Results


## Results Table

| Corruption $\rho$ | Description | Ant U-Maze | Ant Soccer |
| :---: | :--- | :---: | :---: |
| 0.00 | Full GCHR (trajectory-aligned goals) | 0.327 $\pm$ 0.083 | 0.463 $\pm$ 0.022 |
| 0.25 | 25% random replay-buffer goals | 0.298 $\pm$ 0.065 | 0.417 $\pm$ 0.037 |
| 0.50 | 50% random replay-buffer goals | 0.267 $\pm$ 0.059 | 0.383 $\pm$ 0.060 |
| 0.75 | 75% random replay-buffer goals | 0.276 $\pm$ 0.057 | 0.378 $\pm$ 0.099 |
| 1.00 | All random goals (no trajectory alignment) | 0.243 $\pm$ 0.083 | 0.362 $\pm$ 0.095 |
| -- | SAC+HER (no GCHR) | 0.300 $\pm$ 0.027 | 0.001 $\pm$ 0.002 |

<p align="center">
  <img src="goal_corruption_combined.png" width="100%" />
  <br>
  <em>Figure 1: Final success rate as a function of goal corruption fraction ρ.</em>
</p>

<p align="center">
  <img src="learning_curves_combined.png" width="100%" />
  <br>
  <em>Figure 2: Learning curves across corruption levels. Clean GCHR (ρ=0) learns faster and reaches higher asymptotic performance than fully corrupted (ρ=1) in both environments.</em>
</p>

