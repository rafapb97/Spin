import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import os

#import pyNN.nest as pynn
import pyNN.spiNNaker as pynn


#set sim parameters
sim_time = 200
dt = 0.1
refrac = 0
start_test=0
end_test= 4
print("trials from "+str(start_test)+ "to "+str(end_test))
pynn.setup(dt)
pynn.set_number_of_neurons_per_core(pynn.IF_curr_exp, 64)
weight_scale = 1
rescale_fac = 1000/(1000*dt)

#load data
test_data = np.load("x_test.npz")['arr_0'][start_test:end_test]
test_labels = np.load("y_test.npz")['arr_0'][start_test:end_test]

pred_labels = []

#create network
network = []


#cell defaults
cell_params = {
'v_thresh' : 0.7,
'tau_refrac' : refrac,
'v_reset' : 0,
'v_rest' : 0,
'cm' : 1,
'tau_m' : 1000,
'tau_syn_E' : 0.01,
'tau_syn_I' : 0.01
}

#create populations
layer1 = pynn.Population(1024, pynn.SpikeSourcePoisson(), label='InputLayer')
#layer1.record("spikes")
network.append(layer1)

layer2 = pynn.Population(3136, pynn.IF_curr_exp, cell_params, label='Conv1')
#layer2.record("spikes")
network.append(layer2)

layer3 = pynn.Population(4608, pynn.IF_curr_exp, cell_params, label='Conv2')
#layer3.record("spikes")
network.append(layer3)

layer4 = pynn.Population(1152, pynn.IF_curr_exp, cell_params, label='Conv3')
#layer4.record("spikes")
network.append(layer4)

layer5 = pynn.Population(5, pynn.IF_curr_exp, cell_params, label='Output')
layer5.record("spikes")
network.append(layer5)

spiketrains_all=[]

#create connections
filenames=[
    "0Conv2D_14x14x16",
    "1Conv2D_12x12x32",
    "2Conv2D_12x12x8",
    "4Dense_5"
]



for i in range(len(network)-1):
    ex = np.genfromtxt(filenames[i]+"_excitatory")
    inh = np.genfromtxt(filenames[i]+"_inhibitory")
    ex[:,2] /= weight_scale 
    inh[:,2] /= weight_scale 
    pynn.Projection(network[i], network[i+1], pynn.FromListConnector(ex, ['weight', 'delay']), receptor_type='excitatory')
    pynn.Projection(network[i], network[i+1], pynn.FromListConnector(inh, ['weight', 'delay']), receptor_type='inhibitory')


    #network[i+1].initialize(v=0.0, isyn_exc=0.0, isyn_inh=0.0)
    #'isyn_exc': 0.0, 'isyn_inh': 0.0,
#set input
firstrun = True

for trial, j in enumerate(test_data):
    if firstrun:
        firstrun = False
    else:
        pynn.reset()
    try:
        for i in range(len(network)-1):
            network[i+1].initialize(v=0.0)
                
        x_flat = np.ravel(j)
        rates = 1000 * x_flat / rescale_fac
        network[0].set(rate=rates)

        #run simulation
        pynn.run(sim_time)

        #get spikes
        shape = (5, int(sim_time/dt))
        spiketrains = network[-1].get_data().segments[-1].spiketrains
        spiketrains_flat = np.zeros(shape)
        for k, spiketrain in enumerate(spiketrains):
            for t in spiketrain:
                spiketrains_flat[k, int(t / dt)] = 1

        spikesum = np.sum(spiketrains_flat, axis = 1)

        pred_labels.append(np.eye(5)[np.argmax(spikesum)])

        print(spikesum)
        print('estimate = ' + str(np.argmax(spikesum)))

    except:
        print("COULD NOT RUN TRIAL " + str(trial))
        pred_labels.append(np.zeros(5))

   


#get spikes for plotting
spiketrains_all.append(spiketrains_flat)
    

#end simulation
pynn.end()




print('simulation end')

s = []
a = []
i = 0
for spiketrain in spiketrains_all:
    eventdata = []
    for dat in spiketrain:
        eventdata.append(np.nonzero(dat)[0])
    plt.figure()
    plt.eventplot(eventdata)
    plt.xlim(0, int(sim_time / dt))
    plt.ylim(-0.5, 5.5)
    plt.xlabel("timesteps (0.1 ms)")
    plt.ylabel("neuron no")
    plt.savefig("spikes trial: " + str(i) + ".png")
    i+=1

print('loop end')


good_preds=0.0
for i in range(len(pred_labels)):
    good_preds +=np.dot(pred_labels[i], test_labels[i])
print("accuracy: "+str(good_preds/(i+1)))
