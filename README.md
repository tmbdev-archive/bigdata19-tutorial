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
