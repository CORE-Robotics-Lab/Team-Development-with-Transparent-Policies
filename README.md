# Designs for Enabling Collaboration in
Human-Machine Teaming via Interactive and
Explainable Systems

This is the codebase for 
"Designs for Enabling Collaboration in
Human-Machine Teaming via Interactive and
Explainable Systems," which is published in NeurIPS 2024 (poster).
The presentation video of this work can be found here [FILL]. 

Authors: [Rohan Paleja](rohanpaleja.com), Michael Munje, Kimberlee Chang, Reed Jensen, and Matthew Gombolay

### Training Agent Models
As mentioned in the paper, we leverage the PantheonRL codebase to train agents and change the agent representation to be an Interpretable
Discrete Control Tree instead of the default Neural Network. The IDCT model can be found in ipm/models/idct.py and the domains 
can be found in overcooked_ai/src/overcooked_ai_py/data/layouts/two_rooms_narrow.layout and overcooked_ai/src/overcooked_ai_py/data/layouts/forced_coordination.layout

### Running the interactive policy modification GUI
The main file of this experiment is run_interactive_experiment.py. This file runs the human-subjects study while accounting for 
which condition the user is in. This file handles all rendering of the Overcooked-AI domain as well as the tree visualization and 
modification interface.

