import os
import glob
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

def create_hparam_config(config_dir):
    # Step 1: Clean old config (delete all .tfevents files)
    event_files = glob.glob(os.path.join(config_dir, "events.out.tfevents.*"))
    for f in event_files:
        os.remove(f)
        print(f"[HParams Config] Removed old config file: {f}")

    print(f"[HParams Config] Writing new config to: {config_dir}")

    # === Define HParams ===
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([32]))
    HP_Z_DIM = hp.HParam('z_dim', hp.Discrete([128]))
    HP_GF_DIM = hp.HParam('gf_dim', hp.Discrete([64]))
    HP_DF_DIM = hp.HParam('df_dim', hp.Discrete([64]))
    HP_G_LR = hp.HParam('generator_lr', hp.Discrete([1e-3, 5e-4]))
    HP_D_LR = hp.HParam('discriminator_lr', hp.Discrete([1e-3, 5e-4]))
    HP_C_CYCLE = hp.HParam('critic_cycle', hp.Discrete([1, 2, 3, 4, 5]))
    HP_C_CYCLE_SCH = hp.HParam('critic_cycle_schedule', hp.Discrete([20000]))
    HP_ATTN_POS = hp.HParam('attention_head_position', hp.Discrete([2, 3]))

    writer = tf.summary.create_file_writer(config_dir)
    with writer.as_default():
        hp.hparams_config(
            hparams=[HP_BATCH_SIZE, HP_Z_DIM, HP_GF_DIM, HP_DF_DIM,
                     HP_G_LR, HP_D_LR, HP_C_CYCLE, HP_C_CYCLE_SCH, HP_ATTN_POS],
            metrics=[
                hp.Metric('Generator_Loss', display_name='Gen Loss'),
                hp.Metric('Discriminator_Loss', display_name='Disc Loss'),
            ]
        )
