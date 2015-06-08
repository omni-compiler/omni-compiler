#!/bin/sh

## Check arguments
if test -z $1; then
    echo "make remote-test (user name)"
    exit 1
else
    USER=$1
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
JOBS=${LOCAL_TMP_DIR}/jobs

## Clean Temporal files and dirs
clean_files(){
    echo -n "Clean temporal files... "
    rm -rf ${LOCAL_TMP_DIR}
    ssh ${REMOTE_HOST} "rm -rf ${REMOTE_TMP_DIR}"
    echo "done"
}

cancel_jobs(){
    GET_LAST_JOBS_CMD="squeue -o %A | grep \"^[0-9]\""
    ssh ${REMOTE_HOST} ${GET_LAST_JOBS_CMD} > ${JOBS}
    for id in $(cat ${JOBS}); do
	JOB_CANCEL_CMD="scancel $id"
	ssh -n ${REMOTE_HOST} ${JOB_CANCEL_CMD}
	echo ${JOB_CANCEL_CMD}
    done
}

trap "clean_files" 0
trap "cancel_jobs" 1 2 3 15

## Create archive of the current omni-compiler
echo -n "Compress ... "
if test -d ${LOCAL_TMP_DIR}; then
    echo "Error ! ${LOCAL_TMP_DIR} exist"
    exit 1
else
    mkdir -p ${LOCAL_TMP_DIR}
    cd ..; tar cfj ${LOCAL_TMP_DIR}/${ARCHIVE} ${OMNI}
fi
echo "done"

## Transfer the current omni-compiler
echo -n "Transfer archive ... "
MKDIR_CMD="if test -d ${REMOTE_TMP_DIR}; then; \
             echo \"Error ${REMOTE_HOST}:${REMOTE_TMP_DIR} exist\"; exit 1;\
           else\
             mkdir -p ${REMOTE_TMP_DIR}; \
           fi"
ssh ${REMOTE_HOST} ${MKDIR_CMD}
if test $? -ne 0; then
    exit 1
fi
scp ${LOCAL_TMP_DIR}/${ARCHIVE} ${REMOTE_HOST}:${REMOTE_TMP_DIR}
echo "done"

## Expand the current omni-compiler
echo -n "Expand archive ..."
EXPAND_CMD="tar xfj ${REMOTE_TMP_DIR}/${ARCHIVE} -C ${REMOTE_TMP_DIR}"
ssh ${REMOTE_HOST} ${EXPAND_CMD}
if test $? -ne 0; then
    exit 1
fi
echo "done"

## Compile the current omni-compiler
echo -n "Compile the Omni compiler ..."
COMPILE_CMD="mkdir ${XMP_PATH}; \
             cd ${REMOTE_TMP_DIR}/${OMNI}; \
             sh autogen.sh; \
             ./configure --prefix=${XMP_PATH} --with-gasnet=${GASNET_PATH}; \
             make -j2; make install"
ssh ${REMOTE_HOST} ${COMPILE_CMD}
if test $? -ne 0; then
    exit 1
fi
echo "done"

## Run tests
echo "Run tests ..."
RUN_TESTS_CMD="cd ${REMOTE_TMP_DIR}/${OMNI}; \
               make slurm XMP_PATH=${XMP_PATH}; \
               make slurm-check"
ssh ${REMOTE_HOST} ${RUN_TESTS_CMD}
if test $? -ne 0; then
    exit 1
fi
echo "done"

exit 0
