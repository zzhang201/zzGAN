import tensorflow as tf
import importlib
from protein.protein import Protein
from wgan.model import WGAN
from sngan.generator_gumbel import GumbelGenerator

def get_model(flags, logdir, noise):
    if flags.model_type == "wgan":
        if flags.architecture == "gumbel":
            protein_model = Protein(flags, logdir)
            # TODO: implement toggle for gumbel
        else:
            protein_model = Protein(flags, logdir)
        return WGAN(protein_model, noise)

    elif flags.model_type == "sngan":
        raise NotImplementedError("SNGAN support not yet added back.")

    raise NotImplementedError("Unknown model_type or architecture.")


def get_specific_hooks(flags, logdir):
    hooks = []
    return hooks
