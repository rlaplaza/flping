#!/bin/bash


rootname=${1%%.xyz}
sp=${rootname}.com

if [ "$rootname" = "neutral" ]; then
  charge="0"
fi
if [ "$rootname" = "with_proton" ]; then
  charge="1"
fi
if [ "$rootname" = "with_hydride" ]; then
  charge="-1"
fi


natoms=$(head -n 1 ${1})
nlines=$(wc -l ${1} | cut -d ' ' -f 1)
empty=$(echo " ${nlines} - ${natoms} -2 " | bc -l )
skip=$((${empty%%.*} + 3))
xyz=$(cat ${1} | cut --complement -c60-70 | tail -n +${skip})
pre1="%nprocshared=24"
pre2="%mem=24GB"
pre3="%chk=${rootname##*/}.chk"
route1='#p opt freq=noraman def2svp pbe1pbe em=gd3bj integral=ultrafinegrid'
linker="--Link1--"
route2="#p def2tzvp pbe1pbe integral=ultrafinegrid geom=allcheck guess=read em=gd3bj"

echo "${pre1}" > ${sp}
echo "${pre2}" >> ${sp}
echo "${pre3}" >> ${sp}
echo "${route1}" >> ${sp}
echo "" >> ${sp}
echo "${rootname}" >> ${sp}
echo "" >> ${sp}
echo "${charge} 1" >> ${sp}
echo "${xyz}" >> ${sp}
echo "" >> ${sp}
echo "${linker}" >> ${sp}
echo "${pre1}" >> ${sp}
echo "${pre2}" >> ${sp}
echo "${pre3}" >> ${sp}
echo "${route2}" >> ${sp}
echo "" >> ${sp}

./g16_4cores_sub.sh ${sp}

