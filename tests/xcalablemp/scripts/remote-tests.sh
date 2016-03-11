#!/bin/bash

## Check arguments
if test -z $1; then
    echo "make remote-test (user name)"
    exit 1
else
    USER=$1
    echo "User name is ${USER}"
fi

## Set static arguments
PWD=`pwd`
OMNI=`basename ${PWD}`
PID=$$
LOCAL_TMP_DIR=/tmp/tmp.${PID}
REMOTE_TMP_DIR=/home/${USER}/.tmp.${PID}
ARCHIVE=archive.tar.bz2
REMOTE_HOST=${USER}@omni-compiler.org
XMP_PATH=${REMOTE_TMP_DIR}/work
GASNET_PATH=/opt/GASNet-1.24.2
#JOBS=${LOCAL_TMP_DIR}/jobs

## Clean Temporal files and dirs
clean_files(){
    echo "Clean temporal files ... "
    rm -rf ${LOCAL_TMP_DIR}
    ## Only once "rm -rf", rarely all files are not deleted, because slurm output files are created.
    ## So "rm -rf" are executed twice.
    ssh ${REMOTE_HOST} "rm -rf ${REMOTE_TMP_DIR} 2> /dev/null; rm -rf ${REMOTE_TMP_DIR}"
    exit 0
}

trap "clean_files" 1 2 3 15

omni_exec(){
    ${@+"$@"}
    if test $? -ne 0; then
	exit 1
    fi
}

## Create archive of the current omni-compiler
echo "Compress ... "
if test -d ${LOCAL_TMP_DIR}; then
    echo "Error ! ${LOCAL_TMP_DIR} exist"
    exit 1
else
    mkdir -p ${LOCAL_TMP_DIR}
    cd ..
    omni_exec cp -a ${OMNI} ${LOCAL_TMP_DIR}/${OMNI}
    cd ${LOCAL_TMP_DIR}/${OMNI}
    make clean-tests > /dev/null
    make clean > /dev/null
    cd ..
    omni_exec tar cfj ${ARCHIVE} ${OMNI}
fi

## Transfer the current omni-compiler
echo "Transfer archive ... "
MKDIR_CMD="if test -d ${REMOTE_TMP_DIR}; then \
             echo \"Error ${REMOTE_HOST}:${REMOTE_TMP_DIR} exist\"; exit 1;\
           else\
             mkdir -p ${REMOTE_TMP_DIR}; \
           fi"
omni_exec ssh ${REMOTE_HOST} ${MKDIR_CMD}
omni_exec scp ${LOCAL_TMP_DIR}/${ARCHIVE} ${REMOTE_HOST}:${REMOTE_TMP_DIR}

## Expand the current omni-compiler
echo "Expand archive ..."
EXPAND_CMD="tar xfj ${REMOTE_TMP_DIR}/${ARCHIVE} -C ${REMOTE_TMP_DIR}"
omni_exec ssh ${REMOTE_HOST} ${EXPAND_CMD}

## Compile the current omni-compiler
echo "Compile the Omni compiler ..."
COMPILE_CMD="mkdir ${XMP_PATH}; \
             cd ${REMOTE_TMP_DIR}/${OMNI}; \
             sh autogen.sh; \
             ./configure --prefix=${XMP_PATH} --with-gasnet=${GASNET_PATH}; \
             make -j2; make install"
omni_exec ssh ${REMOTE_HOST} ${COMPILE_CMD}

## Run tests
echo "Run tests ..."
SORTED_LIST=/tmp/xmp_sorted.$$
RUN_TESTS_CMD="cd ${REMOTE_TMP_DIR}/${OMNI}; \
               make slurm XMP_PATH=${XMP_PATH} SORTED_LIST=${SORTED_LIST}"
cd ${LOCAL_TMP_DIR}
omni_exec ssh ${REMOTE_HOST} ${RUN_TESTS_CMD}

# Note : The cancel_jobs() often leaves jobs.
#        When REMOTE_TMP_DIR is removed, the jobs are also removed.
#
#omni_exec ssh ${REMOTE_HOST} ${RUN_TESTS_CMD} | tee ${JOBS}
#
#cancel_jobs(){
#    for id in $(awk '{print $4}' ${JOBS}); do
#	JOB_CANCEL_CMD="scancel $id"
#	ssh -n ${REMOTE_HOST} ${JOB_CANCEL_CMD}
#	echo ${JOB_CANCEL_CMD}
#    done
#    clean_files
#}
#
# trap "cancel_jobs" 1 2 3 15

CHECK_TESTS_CMD="cd ${REMOTE_TMP_DIR}/${OMNI};\
                 make slurm-check;\
                 rm -f $SORTED_LIST"
omni_exec ssh ${REMOTE_HOST} ${CHECK_TESTS_CMD}
echo "done"
echo "PASS ALL TESTS"

clean_files
exit 0
