PatternZero is a reinforcement learning (RL) framework designed to improve **numerical sequence pattern recognition**. By fine-tuning a **small language model (Qwen 2.5)** with **Proximal Policy Optimization (PPO)** and **curriculum learning**, PatternZero aims to enhance mathematical reasoning in small-scale LLMs.

### Key Features
- **Synthetic Dataset Generation**: 20,000+ sequences across **5 difficulty tiers** (linear, recursive, modular, etc.).
- **Reinforcement Learning**: Training with **PPO, GRPO** via veRL and Stable-Baselines3.
- **Fine-tuning LLMs**: Experimentation with **Qwen 2.5 (0.5B & 1.5B)** models.
- **Curriculum Learning**: Gradual difficulty progression for stable training.
- **Performance Evaluation**: Benchmarks against DeepSeek R1 and TinyZero.
