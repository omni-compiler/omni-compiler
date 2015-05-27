#!/bin/sh

SORTED_LIST=$1
BASE_TESTDIR=$2

for subdir in `awk '{print $2}' $SORTED_LIST`; do
    subdir=${subdir%/}
    BFILE=${BASE_TESTDIR}/$subdir/slurm.sh
    DIR=`echo ${BASE_TESTDIR}/$subdir | sed "s/\//\\\\\\\\\//g"`
    sed "s/@DIRNAME@/$DIR/" ./tests/xcalablemp/scripts/slurm_template.sh > $BFILE
    JOBNAME=`echo $subdir | awk -F/ '{print $(NF-1)"-"$NF}'`
    sed "s/@JOBNAME@/$JOBNAME/" $BFILE > $BFILE-2; mv $BFILE-2 $BFILE
    sbatch $BFILE
done
    
