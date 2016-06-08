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
LOCAL_TMP_DIR=/tmp/omnicompiler-remote-test.${PID}
REMOTE_TMP_DIR=${LOCAL_TMP_DIR}
REMOTE_GASNET_DIR=${REMOTE_TMP_DIR}/gasnet
REMOTE_MPI3_DIR=${REMOTE_TMP_DIR}/mpi3
ARCHIVE=archive.tar.bz2
REMOTE_HOST=${USER}@omni.riken.jp
XMP_GASNET_PATH=${REMOTE_GASNET_DIR}/work
XMP_MPI3_PATH=${REMOTE_MPI3_DIR}/work
GASNET_PATH=/opt/GASNet
MPI3_PATH=/opt/openmpi

## Clean Temporal files and dirs

function clean_files()
{
    echo "Clean temporal files ... "
    rm -rf ${LOCAL_TMP_DIR}
    ssh ${REMOTE_HOST} "rm -rf ${REMOTE_TMP_DIR} 2> /dev/null"
}

function omni_exec()
{
    ${@+"$@"}
    if test $? -ne 0; then
	clean_files
	exit 1
     fi
}

## Create archive of omni-compiler
echo "Compress ... "
if test -d ${LOCAL_TMP_DIR}; then
    echo "Error ! ${LOCAL_TMP_DIR} exist"
    exit 1
else
    omni_exec mkdir -p ${LOCAL_TMP_DIR}
    omni_exec cd ..
    omni_exec omni_exec cp -a ${OMNI} ${LOCAL_TMP_DIR}/${OMNI}
    omni_exec cd ${LOCAL_TMP_DIR}/${OMNI}
    omni_exec make clean-tests > /dev/null
    omni_exec make clean > /dev/null
    omni_exec cd ..
    omni_exec tar cfj ${ARCHIVE} ${OMNI}
fi

## Transfer omni-compiler
echo "Transfer archive ... "
CMD="if test -d ${REMOTE_TMP_DIR}; then \
         echo \"Error ${REMOTE_HOST}:${REMOTE_TMP_DIR} exist\"; exit 1;\
     else\
         mkdir -p ${REMOTE_TMP_DIR}; \
     fi"
omni_exec ssh ${REMOTE_HOST} ${CMD}
omni_exec scp ${LOCAL_TMP_DIR}/${ARCHIVE} ${REMOTE_HOST}:${REMOTE_TMP_DIR}

## Expand omni-compiler
echo "Expand archive ..."
CMD="mkdir -p ${REMOTE_GASNET_DIR}; \
     tar xfj ${REMOTE_TMP_DIR}/${ARCHIVE} -C ${REMOTE_GASNET_DIR};\
     mkdir -p ${REMOTE_MPI3_DIR}; \
     tar xfj ${REMOTE_TMP_DIR}/${ARCHIVE} -C ${REMOTE_MPI3_DIR}"
omni_exec ssh ${REMOTE_HOST} ${CMD}

## GASNet
echo ""
echo "-----------------------------------"
echo "  Test omni compiler with GASNet  "
echo "-----------------------------------"
CMD="export PATH=${MPI3_PATH}/bin:${XMP_GASNET_PATH}/bin:$PATH && \
     cd ${REMOTE_GASNET_DIR}/${OMNI} && \
     sh autogen.sh && \
     ./configure --prefix=${XMP_GASNET_PATH} --with-gasnet=${GASNET_PATH} && \
     make -j && make install && make tests -j && make run-tests"
omni_exec ssh ${REMOTE_HOST} ${CMD}

## MPI3
echo ""
echo "-----------------------------------"
echo "   Test omni compiler with MPI3   "
echo "-----------------------------------"
CMD="export PATH=${MPI3_PATH}/bin:${XMP_MPI3_PATH}/bin:$PATH && \
     cd ${REMOTE_MPI3_DIR}/${OMNI} && \
     sh autogen.sh && \
     ./configure --prefix=${XMP_MPI3_PATH} && \
     make -j && make install && make tests -j && make run-tests"
omni_exec ssh ${REMOTE_HOST} ${CMD}

echo ""
echo "PASS ALL TESTS"

clean_files
exit 0
