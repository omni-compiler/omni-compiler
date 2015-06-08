#!/bin/sh

BASE_TESTDIR=$1
shift
TESTDIRS=$@
INTERVAL=10

# Count number of test directories
ALL_JOBS=0
for subdir in ${TESTDIRS}; do
    ALL_JOBS=`expr ${ALL_JOBS} + 1`
done

ELAPSED_TIME=0
echo "Waiting to finish jobs ..."
while : ; do
    sleep ${INTERVAL}
    ELAPSED_TIME=`expr ${ELAPSED_TIME} + ${INTERVAL}`
    ALL_EXIST=true
    DONE_JOBS=0
    for subdir in ${TESTDIRS}; do
	EFILE=${BASE_TESTDIR}/${subdir}/slurm_error_code
	if test -e ${EFILE}; then
	    ERR_CODE=`cat $EFILE`
	    if test ${ERR_CODE} -ne 0; then
		cat ${BASE_TESTDIR}/${subdir}/log
		cat ${BASE_TESTDIR}/${subdir}/err
		exit $ERR_CODE
	    fi
	    DONE_JOBS=`expr $DONE_JOBS + 1`
	else
	    ALL_EXIST=false
	fi
    done
    echo "Elapse time is ${ELAPSED_TIME} sec. (${DONE_JOBS}/${ALL_JOBS})"
    
    if ${ALL_EXIST}; then
	break
    fi
done
