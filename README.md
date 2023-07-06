
# MASAC IPU

MASIC IPU is a fork of the MASAC repository available at: https://github.com/ffelten/MASAC

The objective of MASIC IPU is to enhance the performance of MASAC by leveraging Graphcore IPU technology.

The updates made to MASIC IPU are relatively minor in comparison to the official branch.

The library dependencies on the Poplar SDK 3.0 are the following :

```bash
# Install experimental JAX for IPUs (SDK 3.1) from Github releases.
import sys
!{sys.executable} -m pip uninstall -y jax jaxlib
!{sys.executable} -m pip install jax==0.3.16+ipu jaxlib==0.3.15+ipu.sdk310 -f https://graphcore-research.github.io/jax-experimental/wheels.html

!pip install --no-deps flax==0.5.0 # recent version of flax: AttributeError: module 'jax.tree_util' has no attribute 'register_pytree_with_keys_class'
!pip install gymnasium==0.28.1
!pip install wandb==0.12.8
!pip install pettingzoo==1.23.1
!pip install pygame==2.5.0
!pip install msgpack==1.0.5
!pip install --no-deps optax==0.1.2
!pip install --no-deps chex==0.1.5
!pip install dm-tree==0.1.5
!pip install toolz==0.10.0
!pip install protobuf==3.19.6
!pip install tensorboard==2.13.0
```

Regrettably, the development is currently on hold as it lacks the advantage when compared to NVIDIA GPUs. Nonetheless, there is potential for IPU to exhibit significantly faster performance in certain applications. We anticipate future advancements in Reinforcement Learning agents and Simulator implementation with FLAX built on top of JAX+IPU.

