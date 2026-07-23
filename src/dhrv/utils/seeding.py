import random
import numpy as np


def set_global_seed(seed: int) -> None:
    """Fixe la seed numpy et random pour reproductibilité.

    À appeler en tout début de notebook/script, avant toute simulation.
    """
    random.seed(seed)
    np.random.seed(seed)