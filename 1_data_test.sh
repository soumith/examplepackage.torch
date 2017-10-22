echo "enter number of nodes"
read nodes

/opt/openmpi-2.0.1/bin/mpirun -n $nodes -npernode 1 --hostfile /home/ubuntu/TorchMPI-master/hostfile --bind-to none /home/ubuntu/torch/install/bin/luajit test/test.lua 