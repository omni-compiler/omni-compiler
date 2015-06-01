#!/bin/sh

BASE_TESTDIR=$1
shift
TESTDIRS=$@
TIME_INTERVAL=10

while : ; do
    ALL_EXIST=true
    sleep $TIME_INTERVAL
    for subdir in ${TESTDIRS}; do
	EFILE=${BASE_TESTDIR}/$subdir/slurm_error_code
	if test -e $EFILE; then
	    ERR_CODE=`cat $EFILE`
	    if test $ERR_CODE -ne 0; then
		cat ${BASE_TESTDIR}/$subdir/log
		cat ${BASE_TESTDIR}/$subdir/err
		exit $ERR_CODE
	    fi
	else
	    ALL_EXIST=false
	fi
    done
    
    if $ALL_EXIST; then
	break
    fi
done
