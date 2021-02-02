import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

import pyNN.spiNNaker as pynn

print (pynn.IF_cond_exp.default_parameters)

#set sim parameters
sim_time = 200
dt = 0.1

#load data
train_data = np.load("x_test.npz")
data = train_data['arr_0'][0]

#setup pynn
pynn.setup(dt)

#create network
network = []

#cell defaults
"""cell_params = {
'v_thresh' : 1,
'tau_refrac' : 0,
'v_reset' : 0,
'v_rest' : 0,
'cm' : 1,
'tau_m' : 1000,
'tau_syn_E' : 0.01,
'tau_syn_I' : 0.01}"""

#create populations

layer1 = pynn.Population(784, pynn.SpikeSourcePoisson(), label='InputLayer')
layer1.record("spikes")
network.append(layer1)


layer2 = pynn.Population(676, pynn.IF_curr_exp)
layer2.record("spikes")
network.append(layer2)

layer3 = pynn.Population(10, pynn.IF_curr_exp)
layer3.record("spikes")
network.append(layer3)

#create connections
#pynn.Projection(input, layer1)

pynn.Projection(layer1, layer2, pynn.FromListConnector(np.genfromtxt("0Conv2D_13x13x4_excitatory")))
pynn.Projection(layer1, layer2, pynn.FromListConnector(np.genfromtxt("0Conv2D_13x13x4_inhibitory")))

pynn.Projection(layer2, layer3, pynn.FromListConnector(np.genfromtxt("2Dense_10_excitatory")))
pynn.Projection(layer2, layer3, pynn.FromListConnector(np.genfromtxt("2Dense_10_inhibitory")))

#set input
x_flat = np.ravel(data)


rescale_fac = 1000/(1000*0.1)
#rescale_fac = 1000 / (self.config.getint('input', 'input_rate') *self._dt)
rates = 1000 * x_flat / rescale_fac
network[0].set(rate=rates)


#run simulation
pynn.run(sim_time)

#get spikes
spikes_brains = list()
for brain in network:
    spikes_brains.append(brain.get_data("spikes").segments[0].spiketrains)

#end simulation
pynn.end()

#generate some plots
for i, spikes_brain in zip(range(len(network)), spikes_brains):
    fig = plt.figure(figsize=(12, 6))
    grid = gs.GridSpec(3, 1, height_ratios=(1, 1, 4))

    ax_spikes = fig.add_subplot(grid[2, 0])

    for (nrn, spike_train) in enumerate(spikes_brain):
        ax_spikes.plot(
            spike_train, np.ones_like(spike_train) * nrn, "|", c='r', ms=5)

    mn_spiketimes = [time for spikes in spikes_brain for time in spikes]
    mn_signal, mn_times = np.histogram(
        mn_spiketimes, bins=np.arange(0.0, sim_time, 3.0))

    ax_spikes.set_xlabel("time [ms] layer" + str(i) )
    ax_spikes.set_ylabel("")

    #ax_spikes.set_yticks(np.arange(0, 10, 1))
    #ax_spikes.set_yticklabels(["", "", "MN", "", "", "", "", "~MN", "", ""])
    ax_spikes.set_xlim(-1, sim_time+1)

    fig.savefig("spikes layer: " + str(i) + ".png")
