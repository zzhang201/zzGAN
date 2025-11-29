import tensorflow as tf
import importlib
from protein.protein import Protein
from wgan.model import WGAN
from sngan.generator_gumbel import GumbelGenerator

def get_model(flags, logdir, noise):
    
    # 1. Configure the Architecture based on flags
    if flags.model_type == "wgan":
        print(">>> CONFIGURING FOR WGAN-GP (LayerNorm=ON, SN=OFF, GP=ON)")
        use_sn = False
        use_ln = True
        lambda_gp = 20.0  # High penalty for stability
        
    elif flags.model_type == "sngan":
        print(">>> CONFIGURING FOR SNGAN (LayerNorm=OFF, SN=ON, GP=OFF)")
        use_sn = True
        use_ln = False
        lambda_gp = 0.0   # No gradient penalty
    else:
        raise NotImplementedError(f"Unknown model_type: {flags.model_type}")

    # 2. Instantiate the wrapper (Protein/WGAN)
    flags.use_sn = use_sn
    flags.use_ln = use_ln
    
    # Create the base model structure
    protein_model = Protein(flags, logdir)

    # 3. Return the Trainer Wrapper
    # Pass the calculated lambda_gp here
    return WGAN(protein_model, noise)

    
def get_specific_hooks(flags, logdir):
    hooks = []
    return hooks
