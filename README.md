code for HH model neuron

This code was used to generate Figure 2.

currents_visualization.py includes currentscape plotting code.

Model was built with NEURON modul, so previously installed NEURON is necessary for the code to work. More information about NEURON installation on https://neuron.yale.edu/neuron/download

Random_spike_number.py contains the model and the current contribution calculation code. Note that the model's input pattern is generated by a Poisson process, so Figure 2 cannot be reproduced exactly.

Also note that currentscape plotting can last for a lot of time depending on the duration of the simulation.

