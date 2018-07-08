Poster
===

The poster can be found in `/doc/poster/DL4CV.pdf`.


Folder-Structure
===

All code is located in `/src`. The `/setup` folder contains some install-scripts which are mainly required for non-linux-environments. The windows setups may reference non-existing resources and may not work.

Requirements
===

Python ≥3.6 is required. All other requirements can be found in `/setup/requirements-linux.txt`.


Running the Code
===

You can train either on CIFAR10, or on MNIST. Run the Python-file respectively with:

`python3 src/cifar.py [-h]`

This will show you all available run-options. They are all optional though, so you can just go ahead and start training with:

`python3 src/cifar.py`

For MNIST, run `mnist.py`.

Semi-Supervised-Learning
---

The classifier can be trained with the jupyter notebook `/src/TrainClassifierOnGAN.ipynb`. Some boxes are for CIFAR10, others for MNIST only.
