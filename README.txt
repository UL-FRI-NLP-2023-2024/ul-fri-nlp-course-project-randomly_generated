You will first need to update your .bash_profile fille on the server.
After that you need Singularity.def in your file location on the server.
next enter this commands 

mkdir containers

##you have to change the path to your Singularity.def file 

singularity build ./containers/project.sif /d/hpc/home/zp68409/Singularity.def

Lastly you will need the run_slurm.sh file 

sbatch run_slurm.sh 
