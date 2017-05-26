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

GASNET_OPENMPI_BASE_DIR=${REMOTE_TMP_DIR}/gasnet-openmpi
GASNET_OPENMPI_INSTALL_DIR=${GASNET_OPENMPI_BASE_DIR}/work
GASNET_MPICH_BASE_DIR=${REMOTE_TMP_DIR}/gasnet-mpich
GASNET_MPICH_INSTALL_DIR=${GASNET_MPICH_BASE_DIR}/work
OPENMPI_BASE_DIR=${REMOTE_TMP_DIR}/openmpi
OPENMPI_INSTALL_DIR=${OPENMPI_BASE_DIR}/work
MPICH_BASE_DIR=${REMOTE_TMP_DIR}/mpich
MPICH_INSTALL_DIR=${MPICH_BASE_DIR}/work

ARCHIVE=archive.tar.bz2
REMOTE_HOST=${USER}@omni.riken.jp
GASNET_OPENMPI_PATH=/opt/GASNet-openmpi
GASNET_MPICH_PATH=/opt/GASNet-mpich
OPENMPI_PATH=/opt/openmpi
MPICH_PATH=/opt/mpich

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
CMD="mkdir -p ${GASNET_OPENMPI_BASE_DIR}; \
     tar xfj ${REMOTE_TMP_DIR}/${ARCHIVE} -C ${GASNET_OPENMPI_BASE_DIR};\
     cp -a ${GASNET_OPENMPI_BASE_DIR} ${GASNET_MPICH_BASE_DIR}; \
     cp -a ${GASNET_OPENMPI_BASE_DIR} ${OPENMPI_BASE_DIR}; \
     cp -a ${GASNET_OPENMPI_BASE_DIR} ${MPICH_BASE_DIR}"
omni_exec ssh ${REMOTE_HOST} ${CMD}

## Output summury of compilers
echo ""
echo "------------------------"
echo "  Summury of compilers  "
echo "------------------------"
CMD="LANG=C; gcc -v 2>&1 > /dev/null | tail -1 && \
     ls -l1 /opt/GASNet-openmpi | awk '{print \$11}' && \
     ls -l1 /opt/GASNet-mpich   | awk '{print \$11}'"
omni_exec ssh ${REMOTE_HOST} ${CMD}

## GASNet and openmpi
echo ""
echo "----------------------------------------------"
echo "  Test omni compiler with GASNet and openmpi  "
echo "----------------------------------------------"
CMD="export PATH=${OPENMPI_PATH}/bin:${GASNET_OPENMPI_INSTALL_DIR}/bin:$PATH && \
     cd ${GASNET_OPENMPI_BASE_DIR}/${OMNI} && \
     sh autogen.sh && \
     ./configure --prefix=${GASNET_OPENMPI_INSTALL_DIR} --with-gasnet=${GASNET_OPENMPI_PATH} && \
     make -j16 && make install && make clean-tests && make tests -j16 && make run-tests"
omni_exec ssh ${REMOTE_HOST} ${CMD}

## GASNet and mpich
echo ""
echo "--------------------------------------------"
echo "  Test omni compiler with GASNet and mpich  "
echo "--------------------------------------------"
CMD="export PATH=${MPICH_PATH}/bin:${GASNET_MPICH_INSTALL_DIR}/bin:$PATH && \
     cd ${GASNET_MPICH_BASE_DIR}/${OMNI} && \
     sh autogen.sh && \
     ./configure --prefix=${GASNET_MPICH_INSTALL_DIR} --with-gasnet=${GASNET_MPICH_PATH} && \
     make -j16 && make install && make clean-tests && make tests -j16 && make run-tests"
omni_exec ssh ${REMOTE_HOST} ${CMD}

## OpenMPI
echo ""
echo "-----------------------------------"
echo "  Test omni compiler with OpenMPI  "
echo "-----------------------------------"
CMD="export PATH=${OPENMPI_PATH}/bin:${OPENMPI_INSTALL_DIR}/bin:$PATH && \
     cd ${OPENMPI_BASE_DIR}/${OMNI} && \
     sh autogen.sh && \
     ./configure --prefix=${OPENMPI_INSTALL_DIR} && \
     make -j16 && make install && make clean-tests && make tests -j16 && make run-tests"
omni_exec ssh ${REMOTE_HOST} ${CMD}

## MPICH
echo ""
echo "---------------------------------"
echo "  Test omni compiler with MPICH  "
echo "---------------------------------"
CMD="export PATH=${MPICH_PATH}/bin:${MPICH_INSTALL_DIR}/bin:$PATH && \
     cd ${MPICH_BASE_DIR}/${OMNI} && \
     sh autogen.sh && \
     ./configure --prefix=${MPICH_INSTALL_DIR} && \
     make -j16 && make install && make clean-tests && make tests -j16 && make run-tests"
omni_exec ssh ${REMOTE_HOST} ${CMD}

echo "------------------"
echo "  PASS ALL TESTS  "
echo "------------------"

clean_files
exit 0
