import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

#import pyNN.nest as pynn

import pyNN.spiNNaker as pynn
#print (pynn.IF_cond_exp.default_parameters)

#set sim parameters
sim_time = 50
dt = 0.1

#load data
test_data = np.load("x_test.npz")['arr_0'][0:10]

test_labels = np.load("y_test.npz")['arr_0'][0:10]
#print("Corr is: " + str(train_labels['arr_0'][0]))



pred_labels = []

for j in test_data:
    #setup pynn
    pynn.setup(dt)
    #pynn.set_number_of_neurons_per_core(pynn.IF_curr_exp, 64)
    #create network
    network = []

    #cell defaults
    cell_params = {
    'v_thresh' : 1,
    'tau_refrac' : 0.1,
    'v_reset' : 0,
    'v_rest' : 0,
    'cm' : 1,
    'tau_m' : 1000,
    'tau_syn_E' : 0.01,
    'tau_syn_I' : 0.01
    }

    #create populations
    #Number of neurons: 6314
    #Number of synapses: 967488

    layer1 = pynn.Population(784, pynn.SpikeSourcePoisson(), label='InputLayer')
    layer1.record("spikes")
    network.append(layer1)

    layer2 = pynn.Population(2304, pynn.IF_curr_exp(**cell_params), label='Conv1')
    layer2.record("spikes")
    network.append(layer2)

    layer3 = pynn.Population(3200, pynn.IF_curr_exp, cell_params, label='Conv2')
    layer3.record("spikes")
    network.append(layer3)

    layer4 = pynn.Population(800, pynn.IF_curr_exp, cell_params, label='Conv3')
    layer4.record("spikes")
    network.append(layer4)

    layer5 = pynn.Population(10, pynn.IF_curr_exp, cell_params, label='Output')
    layer5.record("spikes")
    network.append(layer5)


    #create connections
    #pynn.Projection(input, layer1)
    filenames=[
        "0Conv2D_12x12x16",
        "1Conv2D_10x10x32",
        "2Conv2D_10x10x8",
        "4Dense_10"
    ]
    #FromFileConnector




    weight_scale = 1
    for i in range(len(network)-1):
        ex = np.genfromtxt(filenames[i]+"_excitatory")
        inh = np.genfromtxt(filenames[i]+"_inhibitory")
        ex[:,2] /= weight_scale 
        inh[:,2] /= weight_scale 
        pynn.Projection(network[i], network[i+1], pynn.FromListConnector(ex, ['weight', 'delay']), receptor_type='excitatory')
        pynn.Projection(network[i], network[i+1], pynn.FromListConnector(inh, ['weight', 'delay']), receptor_type='inhibitory')

        network[i+1].initialize(v=0.0)
        #network[i+1].initialize(v=0.0, isyn_exc=0.0, isyn_inh=0.0)
        #'isyn_exc': 0.0, 'isyn_inh': 0.0,
    #set input
    x_flat = np.ravel(j)


    rescale_fac = 1000/(1000*dt)
    #rescale_fac = 1000 / (self.config.getint('input', 'input_rate') *self._dt)
    rates = 1000 * x_flat / rescale_fac
    #print(rates)
    network[0].set(rate=rates)


    #run simulation
    pynn.run(sim_time)

    #get spikes
    shape = [10, int(sim_time/dt)]
    spiketrains = network[-1].get_data().segments[-1].spiketrains
    spiketrains_flat = np.zeros((shape[0], shape[1]))
    for k, spiketrain in enumerate(spiketrains):
        for t in spiketrain:
            spiketrains_flat[k, int(t / dt)] = t

    spiketrains_b_l_t = np.reshape(spiketrains_flat, shape)


    spikesum = np.sum(spiketrains_b_l_t, axis = 1)
    
    pred_labels.append(np.eye(10)[np.argmax(spikesum)])
    
    print(spikesum)
    print('estimate = ' + str(np.argmax(spikesum)))

    #get spikes for plotting
    spikes_brains = list()
    for brain in network:
        spikes_brains.append(brain.get_data("spikes").segments[0].spiketrains)

    #end simulation
    pynn.end()

    
    print('simulation end')

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
        
    print('loop end')


good_preds=0
bad_preds=0

for i in range(len(pred_labels)):
    if (pred_labels[i].all() == test_labels[i].all()):
        good_preds = good_preds+1
    else:
        bad_preds=bad_preds+1
print("accuracy: "+str(good_preds/(good_preds+bad_preds)))
        
        