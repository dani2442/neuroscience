from __future__ import annotations

import random
import typing as t

import numpy as np


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	try:
		import torch

		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(seed)
	except Exception:
		# torch might not be installed in the environment running static analysis
		pass


def make_model(name: str, **kwargs) -> t.Callable:
    name = name.lower()
    if name == "linear":
        from nsc.nsde import LinearSDE
        return LinearSDE(**kwargs)
    elif name in {"mlp", "mlpsde", "mlp_sde"}:
        from nsc.nsde import MLPSDE
        return MLPSDE(**kwargs)
    elif name in {"hopf", "hopf_coupled", "hopf_coupled_sde"}:
        from .hopf import HopfCoupledSDE
        return HopfCoupledSDE(**kwargs)
    raise ValueError(f"Unknown model name: {name}")