import contextlib
import tensorflow as tf

class EMA(tf.Module):
    def __init__(self, decay=0.999, name="ema"):
        super().__init__(name=name)
        self.decay = tf.constant(decay, tf.float32)
        self.shadow_vars = []   # trackable fp32 shadows saved in ckpt
        self.pairs = []         # non-trackable: list of (shadow_var, train_var)
        self._built = False
        self._backup = None

    def build(self, model):
        if self._built:
            return
        self.shadow_vars.clear()
        self.pairs.clear()
        for v in model.trainable_variables:
            if "sn_u_vec" in v.name:   # exclude spectral-norm power vector
                continue
            sv = tf.Variable(tf.cast(v, tf.float32), trainable=False,
                             name=v.name.split(":")[0] + "/ema")
            self.shadow_vars.append(sv)
            self.pairs.append((sv, v))
        self._built = True

    @tf.function
    def update(self, model):
        # update by object reference
        for sv, v in self.pairs:
            sv.assign(self.decay * sv + (1. - self.decay) * tf.cast(v, tf.float32))

    def apply_to(self, model):
        # swap live weights -> shadows (save a backup)
        self._backup = [tf.identity(v) for v in model.trainable_variables]
        # map by position in pairs to avoid name confusion
        vset = set(id(v) for _, v in self.pairs)
        j = 0
        for i, v in enumerate(model.trainable_variables):
            if id(v) in vset:
                sv, vv = self.pairs[j]
                v.assign(tf.cast(sv, v.dtype))
                j += 1

    def restore(self, model):
        if self._backup is None:
            return
        for v, b in zip(model.trainable_variables, self._backup):
            v.assign(b)
        self._backup = None

    @contextlib.contextmanager
    def average_parameters(self, model):
        self.apply_to(model)
        try:
            yield
        finally:
            self.restore(model)
