# INT2-Open-Assessment

## MAIN GOALS:

- Construct a suitable (deep)
neural network architecture, train it on the flowers-102 training set, and then evaluate its
classification accuracy on the official flowers-102 test set.

- Prepare a report in the
style of a short academic research paper, presenting your method and results.

## REPORT TLDR:

**1.2 - Software Framework**: PyTorch

**1.3 - Dataset**:

- Oxford 102 Category Flower Dataset (flowers-102)
website: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/

- loaders for PyTorch
https://pytorch.org/vision/stable/generated/torchvision.datasets.Flowers102.html

**1.4 - NN Architecture**:

- Input : image
- Output: specific class prediction

- Use appropriate layers and loss function, train the network using an optimiser on the DS.

- *Network must be trained from scratch*.

**1.5 - Hardware resources**:

- Network must train within 12 hours, on one GPU and/or CPU (it can be any model).

**1.6 - Performance Evaluation**:

- Report must present a final network for the task, that's usable by others.
- Report must use the official flowers-102 test set and cite the accuracy on the test set.

2 - **Grading**:

**Report (40%)**: Present an overview to your method and results. You must evaluate thoroughly
the performance of the network (in terms of accuracy). You must provide justification for the
choices you make.

**Classification performace (30%)**: Report submission has to clearly state the achieved classification accuracy on the official
flowers-102 test set in the abstract and in the Results & Evaluation section. Marks are awarded
according to the achieved test accuracy of the trained model.

**Self- & peer assessment (30%)**: Fill out an online form to rate both yourself and each of your team members.
