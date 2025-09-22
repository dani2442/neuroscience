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