import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
from utils.plot_utils import plot_prediction
from utils.plot_utils import plot_reference
from models.dcgan import DCGAN
from utils.data_utils import DataLoader

def generate(args):
    # Create a new DCGAN object
    dcgan = DCGAN(config)

    # Load existing model from saved_models folder (you can pass different indexes to see the effect on the generated signal)
    dcgan.load() #loads the last trained generator
    #dcgan.load(500)
    #dcgan.load(1000)
    #dcgan.load(2000)
    #dcgan.load(3000)

    # Create a DataLoader utility object
    data_loader = DataLoader(config)

    #
    # Generate a batch of new fake signals and evaluate them against the discriminator
    #

    # Select a random batch of signals
    signals = data_loader.get_training_batch()

    # Generate latent noise for generator
    noise = dcgan.generate_noise(signals)

    # Generate prediction
    gen_signal = dcgan.generator.predict(noise)

    # Evaluate prediction
    validated = dcgan.critic.predict(gen_signal)

    # Plot and save prediction
    plot_prediction(gen_signal)
    gen_signal = np.reshape(gen_signal, (gen_signal.shape[0],gen_signal.shape[1]))
    np.savetxt('./output/generated_signal.csv', gen_signal, delimiter=",")
	

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EMG-GAN - Generate EMG signals based on pre-trained model')
        
    parser.add_argument('--config_json', '-config', default='configuration.json', type=str,
                        help='configuration json file path')          

    args = parser.parse_args()

    config_file = args.config_json
    with open(config_file) as json_file:
        config = json.load(json_file)

    generate(config)
