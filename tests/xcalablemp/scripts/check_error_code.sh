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
echo "Waiting until finish jobs ..."
while : ; do
    sleep ${INTERVAL}
    ELAPSED_TIME=`expr ${ELAPSED_TIME} + ${INTERVAL}`
    DONE_JOBS=0
    for subdir in ${TESTDIRS}; do
	RET_FILE=${BASE_TESTDIR}/${subdir}/slurm_ret_code
	if test -e ${RET_FILE}; then
	    sync
	    RET_VAL=`cat $RET_FILE`
	    if test ${RET_VAL} -ne 0; then
		cat ${BASE_TESTDIR}/${subdir}/log
		cat ${BASE_TESTDIR}/${subdir}/err
		exit $RET_VAL
	    fi
	    DONE_JOBS=`expr $DONE_JOBS + 1`
	fi
    done
    echo "Elapse time is ${ELAPSED_TIME} sec. (${DONE_JOBS}/${ALL_JOBS})"
    
    if test ${DONE_JOBS} -eq ${ALL_JOBS}; then
	break
    fi
done

