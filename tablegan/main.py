import os
import datetime
import tensorflow as tf
import sys
import mlflow
import wandb
from model import TableGan
from utils import pp, generate_data, show_all_variables

#wandb.login()

flags = tf.app.flags

flags.DEFINE_integer("epoch", 10, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", sys.maxsize, "The size of train images [np.inf]")
flags.DEFINE_integer("y_dim", 13, "Number of unique labels")
flags.DEFINE_integer("batch_size", 500, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 16, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 16, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_par_dir", "checkpoint", "Parent Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("checkpoint_dir", "", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("generate_data", False, "True for visualizing, False for nothing [False]")

flags.DEFINE_float("alpha", 0.5, "The weight of original GAN part of loss function [0-1.0]")
flags.DEFINE_float("beta", 0.5, "The weight of information loss part of loss function [0-1.0]")
flags.DEFINE_float("delta_m", 0.5, "")
flags.DEFINE_float("delta_v", 0.5, "")
flags.DEFINE_string("test_id", "5555", "The experiment settings ID.Affecting the values of alpha, beta, delta_m and delta_v.")
flags.DEFINE_integer("label_col", -1, "The column used in the dataset as the label column (from 0). Used if the Classifer NN is active.")
flags.DEFINE_integer("attrib_num", 0, "The number of columns in the dataset. Used if the Classifer NN is active.")
flags.DEFINE_integer("feature_size", 266, "Size of last FC layer to calculate the Hinge Loss function.")
flags.DEFINE_boolean("shadow_gan", False, "True for loading fake data from samples directory[False]")
flags.DEFINE_integer("shgan_input_type", 0, " Input for Discriminator of shadow_gan. 1=Fake, 2=Test, 3=Train Data")

FLAGS = flags.FLAGS

def main(_):
    a = datetime.datetime.now()

    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    if not os.path.exists(FLAGS.checkpoint_par_dir):
        os.makedirs(FLAGS.checkpoint_par_dir)

    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    test_cases = [
        {'id': 'OI_11_00', 'alpha': 1.0, 'beta': 1.0, 'delta_v': 0.0, 'delta_m': 0.0},
        {'id': 'OI_11_11', 'alpha': 1.0, 'beta': 1.0, 'delta_v': 0.1, 'delta_m': 0.1},
        {'id': 'OI_11_22', 'alpha': 1.0, 'beta': 1.0, 'delta_v': 0.2, 'delta_m': 0.2},
        {'id': 'OI_101_00', 'alpha': 1.0, 'beta': 0.1, 'delta_v': 0.0, 'delta_m': 0.0},
        {'id': 'OI_101_11', 'alpha': 1.0, 'beta': 0.1, 'delta_v': 0.1, 'delta_m': 0.1},
        {'id': 'OI_101_22', 'alpha': 1.0, 'beta': 0.1, 'delta_v': 0.2, 'delta_m': 0.2},
        {'id': 'OI_1001_00', 'alpha': 1.0, 'beta': 0.01, 'delta_v': 0.0, 'delta_m': 0.0},
        {'id': 'OI_1001_11', 'alpha': 1.0, 'beta': 0.01, 'delta_v': 0.1, 'delta_m': 0.1},
        {'id': 'OI_1001_22', 'alpha': 1.0, 'beta': 0.01, 'delta_v': 0.2, 'delta_m': 0.2},
        {'id': 'custom_restrict', 'alpha': 0.3, 'beta': 0.8, 'delta_v': 0.1, 'delta_m': 0.1},
        {'id': 'custom', 'alpha': 0.3, 'beta': 0.8, 'delta_v': 0.0, 'delta_m': 0.0},
        {'id': 'custom_info', 'alpha': 1.0, 'beta': 1.2, 'delta_v': 0.1, 'delta_m': 0.1}
    ]

    found = False
    for case in test_cases:
        if case['id'] == FLAGS.test_id:
            found = True
            FLAGS.alpha = case['alpha']
            FLAGS.beta = case['beta']
            FLAGS.delta_m = case['delta_m']
            FLAGS.delta_v = case['delta_v']
            print(case)

    if not found:
        print("Using OI_11_00")
        FLAGS.test_id = "OI_11_00"
        FLAGS.alpha = 1.0
        FLAGS.beta = 1.0
        FLAGS.delta_m = 0.0
        FLAGS.delta_v = 0.0

    FLAGS.input_height = 16
    FLAGS.input_width = 16
    FLAGS.output_height = 16
    FLAGS.output_width = 16

    if FLAGS.shadow_gan:
        checkpoint_folder = FLAGS.checkpoint_par_dir + '/' + FLAGS.dataset + "/" + 'atk_' + FLAGS.test_id
    else:
        checkpoint_folder = f'{FLAGS.checkpoint_par_dir}/{FLAGS.dataset}/{FLAGS.test_id}'

    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    FLAGS.checkpoint_dir = checkpoint_folder

    pp.pprint(flags.FLAGS.__flags)
    print(FLAGS.y_dim)

    # GPU options
    gpu_options = tf.GPUOptions(allow_growth=True)
    run_config = tf.ConfigProto(gpu_options=gpu_options)

    print("Checkpoint : " + FLAGS.checkpoint_dir)

    with tf.Session(config=run_config) as sess:
        tablegan = TableGan(
            sess,
            input_width=FLAGS.input_width,
            input_height=FLAGS.input_height,
            output_width=FLAGS.output_width,
            output_height=FLAGS.output_height,
            batch_size=FLAGS.batch_size,
            sample_num=FLAGS.batch_size,
            y_dim=FLAGS.y_dim,
            dataset_name=FLAGS.dataset,
            crop=FLAGS.crop,
            checkpoint_dir=FLAGS.checkpoint_dir,
            sample_dir=FLAGS.sample_dir,
            alpha=FLAGS.alpha,
            beta=FLAGS.beta,
            delta_mean=FLAGS.delta_m,
            delta_var=FLAGS.delta_v,
            label_col=FLAGS.label_col,
            attrib_num=FLAGS.attrib_num,
            is_shadow_gan=FLAGS.shadow_gan,
            test_id=FLAGS.test_id
        )

        show_all_variables()
        
        # Initialize MLflow experiment
        mlflow.set_experiment("your_experiment_name")
        mlflow.start_run()

        # Log parameters
        mlflow.log_param("epoch", FLAGS.epoch)
        mlflow.log_param("learning_rate", FLAGS.learning_rate)
        mlflow.log_param("beta1", FLAGS.beta1)

        # Train Part
        if FLAGS.train:
            tablegan.train(config=FLAGS, experiment=mlflow)

        # Test Part
        else:
            if not tablegan.load(FLAGS.checkpoint_dir)[0]:
                raise Exception("[!] Train a model first, then run test mode")

            # Below is codes for visualization
            if FLAGS.shadow_gan:  # using Discriminator sampler for Membership Attack
                OPTION = 5
            else:
                OPTION = 1
            
            #generate_data(sess, tablegan, FLAGS, OPTION)
            generate_data(sess, tablegan, config = FLAGS, option=1, num_samples=680880)  # Generate exactly 100K samples
            print('Time Elapsed: ')

            b = datetime.datetime.now()
            print(b - a)


if __name__ == '__main__':
    tf.app.run()
