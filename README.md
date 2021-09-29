This code accompanies the paper "Spacecraft Robotic Capture and Simultaneous Stabilization Using Deep Reinforcement Learning-based Guidance," submitted for possible publication in the Journal of Guidance, Control, and Dynamics.

A D4PG Implementation for manipulator-enabled spacecraft, tasked with capturing and stabilizing a piece of space debris. Trained entirely in simulation, it was then deployed to the Spacecraft Proximity Operations Testbed experimental facility at Carleton University.

This code uses Tensorflow 1.15.0

To run: `python3 main.py`

The default training run will produce a chaser spacecraft the captures and simultaneously stabilizes a target spacecraft. Videos will be produced as well. Good results should be obtained after roughly 14 days.

A pre-trained model and accompanying videos are found in the `Trained model' folder

A video accompanying this work can be found [here](https://youtu.be/_oWpEH_dalo)
