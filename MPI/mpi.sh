#mpi.sh
#!/bin/sh
#PBS -N test
#PBS -l nodes=master_vir2+master_vir3

pssh -h $PBS_NODEFILE mkdir -p /home/s2111045/lab_mpi 1>&2
scp master:/home/s2111045/lab_mpi/mpi /home/s2111045/lab_mpi
pscp -h $PBS_NODEFILE /home/s2111045/lab_mpi/mpi /home/s2111045/lab_mpi 1>&2
mpiexec -np 4 -machinefile $PBS_NODEFILE /home/s2111045/lab_mpi/mpi