import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from brian2 import *

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0  # Normalize the images

# Convert multiple images to spike trains
def pixel_to_spike_trains(images, threshold=0.5):
    return [np.where(image > threshold, 1, 0) for image in images]

# Generate spike trains for the first 1000 images
spike_trains = pixel_to_spike_trains(x_train[:1000])

# Set up the simulation time
duration = 100*ms

# Define neuron equations (integrate-and-fire)
eqs = '''
dv/dt = (1 - v) / (10*ms) : 1
'''

# Set up the neuron group (10 neurons for simplicity)
G = NeuronGroup(10, eqs, threshold='v > 1', reset='v = 0', method='exact')

# Initialize the network
network = Network(G)
network.store('initial_state')  # Store the initial state of the network

# Loop through each spike train and simulate
for train in spike_trains:
    Poisson_input = PoissonGroup(28*28, rates=train.flatten()*Hz)
    S = Synapses(Poisson_input, G, on_pre='v_post += 0.2')
    S.connect()
    spikemon = SpikeMonitor(G)
    
    network.add(Poisson_input, S, spikemon)
    network.run(duration)
    network.remove(Poisson_input, S, spikemon)
    network.restore('initial_state')  # Restore to initial state for each new simulation

# To visualize the final state of spikes after all images processed
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(spikemon.t/ms, spikemon.i, color='k', marker='.')
plt.xlim(0, duration/ms)  # Set x-axis limits to match the simulation duration
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.title('Spike Raster Plot')
plt.show()
