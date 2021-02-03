import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
#import pyNN.nest as pynn
import pyNN.spiNNaker as pynn

print (pynn.IF_cond_exp.default_parameters)

#set sim parameters
sim_time = 500
dt = 0.1

#load data
train_data = np.load("x_test.npz")
data = train_data['arr_0'][1]
train_labels = np.load("y_test.npz")
print("Corr is: " + str(train_labels['arr_0'][0]))


#setup pynn
pynn.setup(dt)

#create network
network = []

#cell defaults
cell_params = {
'v_thresh' : 1,
'tau_refrac' : 0,
'v_reset' : 0,
'v_rest' : 0,
'cm' : 1,
'tau_m' : 1000,
'tau_syn_E' : 0.01,
'tau_syn_I' : 0.01}

#create populations

layer1 = pynn.Population(13068, pynn.SpikeSourcePoisson(), label='InputLayer')
#layer1.record("spikes")
network.append(layer1)

layer2 = pynn.Population(15376, pynn.IF_curr_exp, cell_params, label='Conv1')
#layer2.record("spikes")
network.append(layer2)

layer3 = pynn.Population(26912, pynn.IF_curr_exp, cell_params, label='Conv2')
#layer3.record("spikes")
network.append(layer3)

layer4 = pynn.Population(6728, pynn.IF_curr_exp, cell_params, label='Conv3')
#layer4.record("spikes")
network.append(layer4)

layer5 = pynn.Population(5, pynn.IF_curr_exp, cell_params, label='Output')
layer5.record("spikes")
network.append(layer5)

#create connections
#pynn.Projection(input, layer1)
filenames=[
    "0Conv2D_31x31x16",
    "1Conv2D_29x29x32",
    "2Conv2D_29x29x8",
    "4Dense_5"
]

weight_scale = 1
for i in range(len(network)-1):
	ex = np.genfromtxt(filenames[i]+"_excitatory")
	inh = np.genfromtxt(filenames[i]+"_inhibitory")
	ex[:,2] /= weight_scale 
	inh[:,2] /= weight_scale 
	pynn.Projection(network[i], network[i+1], pynn.FromListConnector(ex, ['weight', 'delay']), receptor_type='excitatory')
	pynn.Projection(network[i], network[i+1], pynn.FromListConnector(inh, ['weight', 'delay']), receptor_type='inhibitory')

        network[i+1].initialize(v=0.0)
#set input
x_flat = np.ravel(data)


rescale_fac = 1000/(1000*dt)
#rescale_fac = 1000 / (self.config.getint('input', 'input_rate') *self._dt)
rates = 1000 * x_flat / rescale_fac
#print(rates)
network[0].set(rate=rates)


#run simulation
pynn.run(sim_time)

#get spikes
spikes_brains = list()
segments=[]
for brain in network:
    spikes_brains.append(brain.get_data("spikes").segments[0].spiketrains)
    segments.append(brain.get_data("spikes").segments[0])

#end simulation
pynn.end()


fig = plt.figure(figsize=(12, 6))
plot_spiketrains(segments[4])
fig.savefig("segment4.png")

#generate some plots
for i, spikes_brain in enumerate(spikes_brains):
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
