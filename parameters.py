from absl import flags
import os

CURRENT_DIRECTORY = os.path.dirname(__file__)
DEFAULT_DATASET = 'GAN/zzGAN/gan/data/original/Length_512_lab_test_37_v2'  # Update as needed

# === Basic setup ===
flags.DEFINE_string('dataset', DEFAULT_DATASET, 'Subdirectory under data_dir where dataset is stored')
flags.DEFINE_string('properties_file', 'properties.json', 'Metadata file inside dataset directory')
flags.DEFINE_string('model_type', 'wgan', 'Model to train: wgan only')
flags.DEFINE_string('logdir', './logs', 'Where logs and checkpoints will be saved')
flags.DEFINE_string('name', 'debug_test_run', 'Run name (used to create subdirectory under logdir)')

flags.DEFINE_string('data_dir', os.path.join(CURRENT_DIRECTORY, '../../data'), 'Base data directory')

# === Training hyperparameters ===
flags.DEFINE_integer('batch_size', 8, 'Batch size for training')
flags.DEFINE_integer('steps', 1000, 'Total training steps')
flags.DEFINE_integer('g_step', 1, 'Number of generator updates per cycle')
flags.DEFINE_integer('d_step', 5, 'Number of discriminator updates per cycle')
flags.DEFINE_integer('shuffle_buffer_size', 1000, 'Buffer size for shuffling dataset')

flags.DEFINE_integer('save_summary_steps', 100, 'Steps between saving summaries')
flags.DEFINE_integer('save_checkpoint_sec', 300, 'Seconds between saving checkpoints')

# === Optimizers ===
flags.DEFINE_float('generator_learning_rate', 1e-4, 'Generator learning rate')
flags.DEFINE_float('discriminator_learning_rate', 1e-4, 'Discriminator learning rate')
flags.DEFINE_float('beta1', 0.5, 'Adam optimizer β1')
flags.DEFINE_float('beta2', 0.9, 'Adam optimizer β2')

# === Model dimensions ===
flags.DEFINE_integer('z_dim', 128, 'Latent noise dimension')
flags.DEFINE_integer('gf_dim', 16, 'Generator base feature dimension')
flags.DEFINE_integer('df_dim', 16, 'Discriminator base feature dimension')

# === Protein-specific settings ===
flags.DEFINE_integer('embedding_height', 58, 'Height of amino acid embedding')
flags.DEFINE_string('embedding_name', 'prot_fp', 'Name of protein embedding')
flags.DEFINE_bool('static_embedding', False, 'Use static (pre-computed) embeddings')
flags.DEFINE_bool('already_embedded', False, 'If inputs are already embedded')
flags.DEFINE_string('pooling', 'conv', 'Pooling type [conv, avg, subpixel]')
flags.DEFINE_integer('dilation_rate', 2, 'Dilation rate for convolutions')

# === Optional regularization ===
flags.DEFINE_float('label_noise_level', 0.0, 'Noise to apply to real/fake labels')
flags.DEFINE_float('noise_level', 0.0, 'Noise added to input tensors')

# === Optional (ignored for now) ===
flags.DEFINE_bool('is_train', True, 'Training mode')
flags.DEFINE_string('running_mode', 'train', 'train or test')

def get_flags():
    return flags.FLAGS
