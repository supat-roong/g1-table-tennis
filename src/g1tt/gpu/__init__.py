"""Self-play training on a GPU: the duel environment in JAX (physics through
MJX, MuJoCo Warp on NVIDIA GPUs) and PPO in PyTorch on the same device.

The environment mirrors g1tt.duel / g1tt.duel_env, so policies and
checkpoints move between the two.
"""
