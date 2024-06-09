#normal.sh
#!/bin/sh
#PBS -N test_normal
#PBS -l nodes=master_vir2

pssh -h $PBS_NODEFILE mkdir -p /home/s2111045/lab_mpi 1>&2
scp master:/home/s2111045/lab_mpi/normal /home/s2111045/lab_mpi
pscp -h $PBS_NODEFILE /home/s2111045/lab_mpi/normal /home/s2111045/lab_mpi 1>&2
/home/s2111045/lab_mpi/normal