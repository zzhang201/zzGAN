import os
import json
import tensorflow as tf


def setup_logdir(flags, properties):
    logdir = os.path.join(flags.logdir, flags.name)
    os.makedirs(logdir, exist_ok=True)

    # Save configs
    with open(os.path.join(logdir, 'config.json'), 'w') as f:
        json.dump(flags.flag_values_dict(), f, indent=4)

    with open(os.path.join(logdir, 'properties.json'), 'w') as f:
        json.dump(properties, f, indent=4)

    return logdir



def print_run_meta_data(flags):
    print("=== Run Parameters ===")
    for k, v in flags.flag_values_dict().items():
        print(f"{k}: {v}")
    print("======================\n")


def get_properties(flags):
    """
    Loads dataset-specific metadata.

    Example contents: number_classes, class_mapping, input shape, etc.
    """
    if flags.dataset.endswith(".json"):
        dataset_path = flags.dataset
    else:
        dataset_path = os.path.join(flags.data_dir, flags.dataset, "properties.json")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset metadata file not found at: {dataset_path}")

    with open(dataset_path, "r") as f:
        return json.load(f)


def add_model_metadata(logdir, model_path, flags, properties):
    """
    Writes a summary note to TensorBoard describing the experiment.
    """
    metadata_summary = f"Model path: {model_path}\n\nFlags:\n{json.dumps(flags.flag_values_dict(), indent=2)}\n\nProperties:\n{json.dumps(properties, indent=2)}"

    writer = tf.summary.create_file_writer(logdir)
    with writer.as_default():
        tf.summary.text(name="run_metadata", data=metadata_summary, step=0)
        writer.flush()


def add_gan_scalars(step, g_loss, d_loss, summary_writer):
    """
    Logs scalar GAN losses to TensorBoard.
    """
    with summary_writer.as_default():
        tf.summary.scalar("1_loss/generator", g_loss, step=step)
        tf.summary.scalar("1_loss/discriminator", d_loss, step=step)
        summary_writer.flush()


# Image visualization no longer needed
# def add_image_grid(...): ❌ DELETED
