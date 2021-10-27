from neuron import h
from neuron.units import ms, mV
import matplotlib.pyplot as plt
from neuron import gui
import numpy as np
from currents_visualization import *
import random
import math
import matplotlib.pyplot as plt
import pandas as pd

h("objref nil")


#Poisson process

rate_ex = 3
rate_inh = 0.1
t_max = 10000 * ms


    
spike_timepoints_ex = []

i = 0
while i < t_max:
    p_ex = random.random()
    inter_spike_time_ex = -math.log(1.0 - p_ex)/rate_ex
    i = i + inter_spike_time_ex
    
    spike_timepoints_ex.append(i)
    
spike_timepoints_ex.pop()



spike_timepoints_inh = []

i = 0
while i < t_max:
    p_inh = random.random()
    inter_spike_time_inh = -math.log(1.0 - p_inh)/rate_inh
    i = i + inter_spike_time_inh
    
    spike_timepoints_inh.append(i)
    
spike_timepoints_inh.pop()    


h.load_file('stdrun.hoc')

#Neuron model

class BallAndStick:
    def __init__(self, gid):
        self._gid = gid
        self._setup_morphology()
        self._setup_biophysics()
    def _setup_morphology(self):
        self.soma = h.Section(name='soma', cell=self)
        self.dend = h.Section(name='dend', cell=self)
        self.all = [self.soma, self.dend]
        #self.dend.connect(self.soma)
        self.soma.L = self.soma.diam = 12.6157
        self.dend.L = 200
        self.dend.diam = 1
    def _setup_biophysics(self):
        for sec in self.all:
            sec.Ra = 100    # Axial resistance in Ohm * cm
            sec.cm = 1      # Membrane capacitance in micro Farads / cm^2
        self.soma.insert('hh')
        for seg in self.soma:
            seg.hh.gnabar = 0.12
            seg.hh.gkbar = 0.036
            seg.hh.gl = 0.0003
            seg.hh.el = -54.3
        self.dend.insert('pas')
        for seg in self.dend:
            seg.pas.g = 0.001
            seg.pas.e = -65
    def __repr__(self):
        return 'BallAndStick[{}]'.format(self._gid)
    
        
my_cell = BallAndStick(0)


#Excitatory Stimulation


syn_ex = h.Exp2Syn(my_cell.soma(0.5))  #netstim
syn_ex.e = 0
syn_ex.tau1 = 0.1 * ms
syn_ex.tau2 = 20 * ms


ncstim_ex = h.NetCon(h.nil,syn_ex)
ncstim_ex.weight[0] = 0.00002  #0.0025 =close to threshold


#Inhibitory stimulation


syn_inh = h.Exp2Syn(my_cell.soma(0.5))
syn_inh.e = -70
syn_inh.tau1 = 0.1 * ms
syn_inh.tau2 = 20 * ms

ncstim_inh = h.NetCon(h.nil,syn_inh)
ncstim_inh.weight[0] = 0.0005   #0.0025 =close to threshold


#cell activity and recording

h.dt = 0.2
dt = 0.2

recording_cell = my_cell
soma_v = h.Vector().record(recording_cell.soma(0.5)._ref_v)


soma_ina = h.Vector().record(recording_cell.soma(0.5)._ref_ina)
soma_ik = h.Vector().record(recording_cell.soma(0.5)._ref_ik)
soma_il = h.Vector().record(recording_cell.soma(0.5).hh._ref_il)

syn_iex = h.Vector().record(syn_ex._ref_i)
syn_iinh = h.Vector().record(syn_inh._ref_i)

apc = h.APCount(recording_cell.soma(0.5))
apc_vector = h.Vector()
apc.record(apc_vector)

t = h.Vector().record(h._ref_t)




def initSpikes_ex():
    for i in spike_timepoints_ex:
        ncstim_ex.event(i)
        
def initSpikes_inh():
    for i in spike_timepoints_inh:
        ncstim_inh.event(i)

fih_ex = h.FInitializeHandler(1, initSpikes_ex)
fih_inh = h.FInitializeHandler(1, initSpikes_inh)

h.finitialize(-65 * mV)

h.continuerun(t_max * ms)


soma_potential = np.array(soma_v)

plt.plot(t, soma_v, label='soma(0.5)')

spike_num = apc.n
# print(spike_num)

plt.legend([spike_num])
plt.show()


#%%
#curentscape plotting

na_current = np.array(soma_ina)
k_current = np.array(soma_ik)
l_current = np.array(soma_il)
syn_iex = np.array(syn_iex)
syn_iinh = np.array(syn_iinh)




soma_potential = np.array(soma_v)
soma_currents = np.array([na_current, k_current, l_current, syn_iex, syn_iinh])


start = 2000 #starting point of currentscape/5

plotCurrentscape(soma_potential[10000:25000], soma_currents[:,10000:25000], spike_timepoints_ex, spike_timepoints_inh, start)

# #intrinsic vs synaptic currents
#%%
cmap = matplotlib.cm.get_cmap('Set1')


intrinsic_curr = sum(abs(na_current) + abs(k_current) + abs(l_current))
syn_curr = sum(abs(syn_iex) + abs(syn_iinh))



data_currents = {'Intrinsic': [intrinsic_curr],
        'Synaptic': [syn_curr]
        }

df_currents = pd.DataFrame(data_currents, columns = ['Intrinsic','Synaptic'])

#print (df_currents)
df_currents.plot.bar(color = [cmap(0.2), cmap(0.4)])


#ex vs. inh
data_synapse = {'Excitatory synapse': [sum(abs(syn_iex))],
        'Inhibitory synapse': [sum(abs(syn_iinh))]
        }

df_synapse = pd.DataFrame(data_synapse, columns = ['Excitatory synapse','Inhibitory synapse'])

#print (df_synapse)
df_synapse.plot.bar(color = [cmap(0.7), cmap(1.0)])


#int currents
data_int = {'iNa': [sum(abs(na_current))],
                'iK': [sum(abs(k_current))],
                'iL': [sum(abs(l_current))]
        }

df_int = pd.DataFrame(data_int, columns = ['iNa', 'iK', 'iL'])

print (df_int)
df_int.plot.bar(color = [cmap(0), cmap(0.3), cmap(0.5)])

#%%
sp =17950
ep =18050
plotCurrentscape(soma_potential[sp:ep], soma_currents[:,sp:ep], spike_timepoints_ex, spike_timepoints_inh, start = sp/5)

#%%
cmap = matplotlib.cm.get_cmap('Set1')

data_int = {'iNa': [sum(abs(na_current[sp:ep]))],
                'iK': [sum(abs(k_current[sp:ep]))]
        }

df_int = pd.DataFrame(data_int, columns = ['iNa', 'iK'])
print (df_int)
df_int.plot.bar(color = [cmap(0), cmap(0.3)])


