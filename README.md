# Deep Learning on Big Data with Multi-Node GPU Jobs

Tutorial held at IEEE BigData 2019

Thomas Breuel, Alex Aizman

Both traditional machine learning (clustering, decision trees, parametric models, cross-validation, function decompositions) and deep learning (DL) are often used for the analysis of big data on hundreds of nodes (clustered servers). However, the systems and I/O considerations for multi-node deep learning are quite different from traditional machine learning. While traditional machine learning is often well served by MapReduce style infrastructure (Hadoop, Spark), distributed deep learning places different demands on hardware, storage software, and networking infrastructure. In this tutorial, we cover: 

- the structure and properties of large-scale GPU-based deep learning systems 
- large-scale distributed stochastic gradient descent and supporting frameworks (PyTorch, TensorFlow, Horovod, NCCL) 
- common storage and compression formats (TFRecord/tf.Example, DataLoader, etc.) and their interconnects (Ethernet, Infiniband, RDMA, NVLINK)
- common storage architectures for large-scale DL (network file systems, distributed file systems, object storage) 
- batch queueing systems, Kubernetes, and NGC for scheduling and large-scale parallelism 
- ETL techniques including distributed GPU-based augmentation (DALI) 

The tutorial will focus on techniques and tools by which deep learning practitioners can take advantage of these technologies and move from single-desktop training to training models on hundreds of GPUs and petascale datasets. It will also help researchers and system engineers to choose and size the systems necessary for such large-scale deep learning. Participants should have some experience in training deep learning models on a single node. The tutorial will cover both TensorFlow and PyTorch frameworks as well as additional open-source tools required to scale deep learning to multi-node storage and multi-node training. 


# Running Jupyter

Many of the examples in this directory are in Jupyter Notebook format.

There are two ways of running Jupyter: on the local machine or inside a container.

To run it directly on the local machine, you need to install Anaconda3.
Afterwads, you can run:

    /opt/anaconda3/bin/jupyter lab

To run Jupyter inside a container, you need to have NVIDIA Docker installed
(e.g., using `ansible-playbook docker-nv.yml`). Then you can use the
`run` script in the parent directory:

    ./run jupyter lab

# Ansible Scripts

These are Ansible scripts that help you set up your machine:

- anaconda3.yml -- install Python3 using Anaconda, plus various packages
- docker-nv.yml -- install Docker with NVIDIA support
- microk8s.yml -- install MicroK8s
- gui.yml -- simple GUI tools for remote access via VNC

These are intended for recent versions of Ubuntu, 19.04 and 19.10. They install
and reinstall various packages, so look at them first before running.

To use:

    $ sudo apt-get install python-pip
    $ sudo pip install ansible
    ...
    $ cd Ansible
    $ ansible-playbook anaconda3.yml
    ...
