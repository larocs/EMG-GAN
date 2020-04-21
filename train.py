import argparse
import json
from models.dcgan import DCGAN
from utils.metrics import *
from utils.plot_utils import plot_losses
from utils.data_utils import DataLoader

def train(config):

        # Create a new DCGAN object
        dcgan = DCGAN(config, training=True)

        # Create a DataLoader utility object
        data_loader = DataLoader(config)

        # Adversarial ground truths
        valid = np.ones((config["batch_size"], 1))
        fake = np.zeros((config["batch_size"], 1))
        
        metrics = []

        for epoch in range(config["epochs"]):

            # Select a random batch of signals
            signals = data_loader.get_training_batch()

            # Generate latent noise for generator
            noise = dcgan.generate_noise(signals)

            # Generate a batch of new fake signals and evaluate them against the discriminator
            gen_signal = dcgan.generator.predict(noise)
            validated = dcgan.critic.predict(gen_signal)
            
            #Sample real and fake signals
                       
            # ---------------------
            #  Calculate metrics
            # ---------------------
            
            # Calculate metrics on best fake data
            metrics_index = np.argmax(validated)

            #Calculate metrics on first fake data
            #metrics_index = 0

            generated = gen_signal[metrics_index].flatten()
            reference = signals[metrics_index].flatten()
            fft_metric, fft_ref, fft_gen = loss_fft(reference, generated)
            dtw_metric = dtw_distance(reference, generated)
            cc_metric = cross_correlation(reference, generated)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            d_loss_real = dcgan.critic.model.train_on_batch(signals, valid) #train on real data
            d_loss_fake = dcgan.critic.model.train_on_batch(gen_signal, fake) #train on fake data
            d_loss = 0.5 * np.add(d_loss_fake, d_loss_real) #mean loss

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = dcgan.combined.train_on_batch(noise, valid) #train combined model
            
            # Plot the progress
            print ("%d [D loss: %f, acc: %f] [G loss: %f] [FFT Metric: %f] [DTW Metric: %f] [CC Metric: %f]" % (epoch, d_loss[0], d_loss[1], g_loss, fft_metric, dtw_metric, cc_metric[0]))
            metrics.append([[d_loss[0]],[g_loss],[fft_metric],[dtw_metric], [cc_metric[0]]])

            # If at save interval => save generated image samples
            if epoch % config["sample_interval"] == 0:
                if config["save_sample"]:
                    dcgan.save_sample(epoch,signals)
                
                if config["plot_losses"]:
                    plot_losses(metrics, epoch)
                
                if config["save_models"]:
                    dcgan.save_critic(epoch)
                    dcgan.save_generator(epoch)
                
        dcgan.save_sample(epoch,signals)
        dcgan.save_critic()
        dcgan.save_generator()
        plot_losses(metrics, epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EMG-GAN - Train')
        
    parser.add_argument('--config_json', '-config', default='configuration.json', type=str,
                        help='configuration json file path')          

    args = parser.parse_args()

    config_file = args.config_json
    with open(config_file) as json_file:
        config = json.load(json_file)

    train(config)
