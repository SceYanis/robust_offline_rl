# Robust Offline RL: Adversarial Attacks and Defenses (PyTorch Implementation)

PyTorch implementation of  
**Towards Robust Policy: Enhancing Offline Reinforcement Learning with Adversarial Attacks and Defenses**  
[[Paper on arXiv]([https://arxiv.org/abs/2405.11206])]  

---

## 📘 Overview

Offline reinforcement learning (RL) enables training policies from pre-collected datasets without costly or risky online exploration. However, this paradigm is vulnerable to **observation perturbations** and **intentional adversarial attacks**, which can degrade policy robustness and real-world performance.

This project proposes a **robust offline RL framework** that:

- 📌 Applies **adversarial attacks** on both the **actor** and **critic** during training by perturbing observations
- 🛡️ Incorporates **adversarial defenses** as regularization strategies to improve policy robustness
- 🧪 Evaluates the framework using the **D4RL benchmark**

---

## 🚀 Features

- ⚔️ 4 types of adversarial attacks (actor-side, critic-side, joint, observation-space)
- 🧠 2 adversarial defenses to mitigate the effects of attacks
- 🔬 Evaluation on D4RL tasks using standard offline RL baselines (e.g., CQL, TD3+BC)
- 🔧 Plug-and-play support for custom attack/defense modules

---

## 🧪 Installation

```bash
git clone https://github.com/thanhkaist/robust_offline_rl.git
cd robust-offline-rl
pip install -r requirements.txt
```


## 📊 Results
Extensive experiments on the D4RL benchmark show:

- Offline RL policies are vulnerable to small adversarial perturbations
- Proposed defenses significantly improve policy robustness across tasks
- Attacks on the critic lead to greater performance drops than on the actor alone

## 📚 Citation
If you find this work useful, please cite:

```bibtex
@inproceedings{nguyen2024towards,
  title={Towards robust policy: Enhancing offline reinforcement learning with adversarial attacks and defenses},
  author={Nguyen, Thanh and Luu, Tung M and Ton, Tri and Yoo, Chang D},
  booktitle={International Conference on Pattern Recognition and Artificial Intelligence},
  pages={310--324},
  year={2024},
  organization={Springer}
}
```
