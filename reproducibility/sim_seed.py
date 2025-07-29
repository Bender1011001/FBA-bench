import random
import numpy as np
from typing import Optional

class SimSeed:
    """
    Manages a global master seed to ensure deterministic reproducibility
    across different random number generators (RNGs) within the simulation.
    
    All RNG sources (NumPy, Python random, any custom volatility calculations)
    must derive their streams from this master seed to ensure complete isolation
    between different RNG streams and overall determinism.
    """
    
    _master_seed: Optional[int] = None
    
    @classmethod
    def set_master_seed(cls, seed: int):
        """
        Sets the global master seed. This should be called once at the start
        of a deterministic simulation run.
        """
        if cls._master_seed is not None:
            raise RuntimeError("Master seed already set. Cannot re-seed during a run.")
        cls._master_seed = seed
        
        # Initialize Python's random module
        random.seed(seed)
        
        # Initialize NumPy's random number generator
        np.random.seed(seed)
        
    @classmethod
    def get_master_seed(cls) -> Optional[int]:
        """
        Returns the current master seed.
        """
        return cls._master_seed

    @classmethod
    def derive_seed(cls, salt: Union[str, int]) -> int:
        """
        Derives a new deterministic seed from the master seed and a given salt.
        This ensures different components get isolated, yet reproducible, streams.
        """
        if cls._master_seed is None:
            raise RuntimeError("Master seed not set. Call set_master_seed() first.")
        
        # Use a combination of the master seed and salt to create a deterministic sub-seed
        # Concatenate string representations of seed and salt, then hash
        combined_string = f"{cls._master_seed}-{salt}"
        return int(hashlib.sha256(combined_string.encode()).hexdigest(), 16) % (2**32 - 1) # Ensure positive int within typical seed range

    @classmethod
    def reset_master_seed(cls):
        """
        Resets the master seed, allowing it to be set again.
        Use with caution, typically only for testing or between simulation runs.
        """
        cls._master_seed = None
        # Reset standard libraries as well to a non-deterministic state or default
        random.seed(None)
        np.random.seed(None) # Resets to a random state based on time or OS


# Initialize the master seed to a non-deterministic state by default on module load
# This ensures that if set_master_seed is not called, we don't accidentally get
# pseudo-deterministic behavior from uninitialized state.
SimSeed.reset_master_seed()