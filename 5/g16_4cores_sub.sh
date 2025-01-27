#!/bin/bash

# We assume running this from the script directory

job_directory=$(pwd)
input=${1}
name="${1%%.com}"
inpname="${name}.com"
outname="${name}.log"
output="${job_directory}/${name}.out"
tmpdir='$SLURM_TMPDIR'
curdir='$SLURM_SUBMIT_DIR'

echo "#!/bin/bash
#SBATCH -J ${name}
#SBATCH -o ${output}
#SBATCH --mem=24000
#SBATCH --nodes=1
#SBATCH -n 24
module load gaussian/g16/C.01
cd ${tmpdir}
echo ${tmpdir}
cp ${curdir}/${inpname} ${tmpdir}
g16 ${inpname} > ${outname}
cp ${outname} ${curdir}
exit " > ${name}.job

sbatch ${name}.job


