from brian2 import *

# Increase simulation duration to 500 ms
duration = 300*ms

# Define neuron equations (integrate-and-fire model)
eqs = '''
dv/dt = (1 - v) / (5*ms) : 1
'''

# Set up the neuron group (10 neurons now)
G = NeuronGroup(10, eqs, threshold='v > 1', reset='v = 0', method='exact')
G.v = 'rand()'  # Initialize membrane potential with random values

# Add Poisson input
Poisson_input = PoissonGroup(10, rates=20*Hz)
S = Synapses(Poisson_input, G, on_pre='v_post += 0.2')
S.connect()

# Set up a spike monitor to record spikes
spikemon = SpikeMonitor(G)

# Run the simulation
run(duration)

# Plot the spikes
plot(spikemon.t/ms, spikemon.i, '.k')
xlabel('Time (ms)')
ylabel('Neuron index')
show()
