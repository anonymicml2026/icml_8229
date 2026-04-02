import glob
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import sys
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)


env = "HandReach"
            #"FetchPick"
            #"HandManipulateBlockRotateZ"

xlim = 200

#if sys.argv[1] == "main":
templates = "(-)rew_monolithic_lr0.001_sd{}.pt"
gcsl_templates = "(-)rew_lr0.001_sd{}.pt"

methods = [
    "ddpg","her","gcsl","wgcsl"
]
colors = ["C0", "C1", "C2", "C3"] 
    #"C4", "C9", "C8"
'''
elif sys.argv[1] == "ablation":
    templates = [
        "her_(-)rew_sym_emb16_lr0.001_sd{}.pt",
        "her_(-)rew_max_emb16_lr0.001_sd{}.pt",
        "her_(-)rew_asym-max-sag-latent_emb16_lr0.001_sd{}.pt",
        "her_(-)rew_asym-max_emb16_lr0.001_sd{}.pt",
    ]
    methods = [
         "MRN (Sym Only)",
         "MRN (Asym Only)",
         "MRN (w/ SAG for e2)",
         "MRN (ours)",
    ]
    colors = ["C6", "C8", "C5", "C9"]
'''
seeds = [100,200,300,400,500]

def smooth(x, delta=2):
    n = x.shape[0]
    b = np.zeros((n,))
    for i in range(n):
        b[i] = x[max(0, i-delta):min(n, i+delta)].mean()
    return b

success = {}
for j, method in enumerate(methods):
    success[method] = []
    for seed in seeds:
        try:
            if method == 'gcsl':
                    filename = f"./results/{env}_{method}_{gcsl_templates.format(seed)}"
            else:
                    filename = f"./results/{env}_{method}_{templates.format(seed)}"
            res = torch.load(filename)
            s = np.array(res['stats']['successes'])
            s = smooth(s)
            success[method].append(s)
        except:
            print("[error] ", env, method, seed)
            continue
    if len(success[method]) > 0:
        min_len = min([len(x) for x in success[method]])
        print(env, method, min_len)
        s = np.stack([x[:min_len] for x in success[method]])
        plt.plot(s.mean(0), color=colors[j], linewidth=3.0, label=method)
        if len(success[method]) > 1:
            plt.fill_between(np.arange(s.shape[1]), s.mean(0) - s.std(0), s.mean(0) + s.std(0), color=colors[j], alpha=0.3)

    env_title = env if "HandManipulate" not in env else env.replace("HandManipulate", "")
    plt.title(env_title, fontsize=20)
    plt.xlim(0, xlim)

    plt.ylim(0, 1.05)

    '''
    plt.spines['top'].set_visible(False)
    plt.spines['right'].set_visible(False)
    plt.spines['bottom'].set_visible(True)
    plt.spines['left'].set_visible(True)

    plt.spines['bottom'].set_edgecolor('black')
    plt.spines['left'].set_edgecolor('black')
    '''

    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    #plt.yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    plt.yticks(np.linspace(0, 1, 21))
    plt.tick_params(which = 'both', direction = 'out')
    

    plt.xlabel("Epoch", fontsize=19)

    plt.ylabel("Success Rate", fontsize=19)
    plt.legend(fontsize=18, loc='lower right')

    plt.grid()

plt.tight_layout()
plt.savefig(env+"alpha_ablation_results.png")
plt.close()
