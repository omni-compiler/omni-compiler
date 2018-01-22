#!/bin/bash -x

## Check arguments
USER=$1
[ "$2" = short ] && SHORT_FLAG=TRUE || SHORT_FLAG=FALSE

## Set static arguments
PWD=`pwd`
OMNI=`basename ${PWD}`
PID=$$
TMP_DIR=/tmp/omni-remote-tests-dir.${PID}
CMD_FILE=${TMP_DIR}/run.sh
STOP_FILE=${TMP_DIR}/stop.sh

GASNET_OPENMPI_BASE_DIR=${TMP_DIR}/gasnet-openmpi
GASNET_OPENMPI_INSTALL_DIR=${GASNET_OPENMPI_BASE_DIR}/work
GASNET_MPICH_BASE_DIR=${TMP_DIR}/gasnet-mpich
GASNET_MPICH_INSTALL_DIR=${GASNET_MPICH_BASE_DIR}/work
OPENMPI_BASE_DIR=${TMP_DIR}/openmpi
OPENMPI_INSTALL_DIR=${OPENMPI_BASE_DIR}/work
MPICH_BASE_DIR=${TMP_DIR}/mpich
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
    echo "Clean temporal dir"
    rm -rf ${TMP_DIR}
    ssh ${REMOTE_HOST} "rm -rf ${TMP_DIR} 2> /dev/null"
}

trap 'ssh ${REMOTE_HOST} "[ -f ${STOP_FILE} ] && sh ${STOP_FILE}"; clean_files; exit 0' 1 2

## Create archive of omni-compiler
echo "Compress ... "
if test -d ${TMP_DIR}; then
    echo "Error ! ${TMP_DIR} exist"
    exit 1
fi

mkdir -p ${TMP_DIR}
echo "Create temporal dir ${TMP_DIR}"
cd ..
cp -a ${OMNI} ${TMP_DIR}/${OMNI}
cd ${TMP_DIR}/${OMNI}
make clean-tests > /dev/null
make clean-tests-F2003 > /dev/null
make clean > /dev/null
rm -rf .git .gitignore
cd ..
tar cfj ${ARCHIVE} ${OMNI}

## Transfer omni-compiler
echo "Transfer archive ... "
CMD="mkdir -p ${TMP_DIR}"
ssh ${REMOTE_HOST} ${CMD}
scp ${TMP_DIR}/${ARCHIVE} ${REMOTE_HOST}:${TMP_DIR}

## Expand omni-compiler
echo "Expand archive ..."
CMD="mkdir -p ${GASNET_OPENMPI_BASE_DIR}; \
     tar xfj ${TMP_DIR}/${ARCHIVE} -C ${GASNET_OPENMPI_BASE_DIR};\
     cp -a ${GASNET_OPENMPI_BASE_DIR} ${GASNET_MPICH_BASE_DIR}; \
     cp -a ${GASNET_OPENMPI_BASE_DIR} ${OPENMPI_BASE_DIR}; \
     cp -a ${GASNET_OPENMPI_BASE_DIR} ${MPICH_BASE_DIR}"
ssh ${REMOTE_HOST} ${CMD}

## Output summury of compilers
echo ""
echo "------------------------"
echo "  Summury of compilers  "
echo "------------------------"
CMD="LANG=C; gcc -v 2>&1 > /dev/null | tail -1 && \
     ls -l1 /opt/GASNet-openmpi | awk '{print \$11}' && \
     ls -l1 /opt/GASNet-mpich   | awk '{print \$11}'"
ssh ${REMOTE_HOST} ${CMD}

cat << EOF > ${CMD_FILE}
PID=\$$
echo "PID is \${PID}"
echo "cd .. ; rm -rf ${TMP_DIR}" > ${STOP_FILE}
echo "kill \${PID}" >> ${STOP_FILE}
echo ""
echo "----------------------------------------------"
echo "  Test omni compiler with GASNet and OpenMPI  "
echo "----------------------------------------------"
export PATH=${OPENMPI_PATH}/bin:${GASNET_OPENMPI_INSTALL_DIR}/bin:$PATH
cd ${GASNET_OPENMPI_BASE_DIR}/${OMNI}
sh autogen.sh
./configure --prefix=${GASNET_OPENMPI_INSTALL_DIR} --with-gasnet=${GASNET_OPENMPI_PATH}
make -j16; make install
make clean-tests;       make tests -j16;       make run-tests
make clean-tests-F2003; make tests-F2003 -j16; make run-tests-F2003

EOF

if test $SHORT_FLAG = FALSE; then
cat << EOF >> ${CMD_FILE}
echo ""
echo "--------------------------------------------"
echo "  Test omni compiler with GASNet and MPICH  "
echo "--------------------------------------------"
export PATH=${MPICH_PATH}/bin:${GASNET_MPICH_INSTALL_DIR}/bin:$PATH
cd ${GASNET_MPICH_BASE_DIR}/${OMNI}
sh autogen.sh
./configure --prefix=${GASNET_MPICH_INSTALL_DIR} --with-gasnet=${GASNET_MPICH_PATH}
make -j16; make install
make clean-tests;       make tests -j16;       make run-tests
make clean-tests-F2003; make tests-F2003 -j16; make run-tests-F2003

## OpenMPI
echo ""
echo "-----------------------------------"
echo "  Test omni compiler with OpenMPI  "
echo "-----------------------------------"
export PATH=${OPENMPI_PATH}/bin:${OPENMPI_INSTALL_DIR}/bin:$PATH
cd ${OPENMPI_BASE_DIR}/${OMNI}
sh autogen.sh
./configure --prefix=${OPENMPI_INSTALL_DIR}
make -j16; make install
make clean-tests;       make tests -j16;       make run-tests
make clean-tests-F2003; make tests-F2003 -j16; make run-tests-F2003

## MPICH
echo ""
echo "---------------------------------"
echo "  Test omni compiler with MPICH  "
echo "---------------------------------"
export PATH=${MPICH_PATH}/bin:${MPICH_INSTALL_DIR}/bin:$PATH
cd ${MPICH_BASE_DIR}/${OMNI}
sh autogen.sh
./configure --prefix=${MPICH_INSTALL_DIR}
make -j16; make install
make clean-tests;       make tests -j16;       make run-tests
make clean-tests-F2003; make tests-F2003 -j16; make run-tests-F2003
EOF
fi

scp ${CMD_FILE} ${REMOTE_HOST}:${CMD_FILE}
ssh ${REMOTE_HOST} sh ${CMD_FILE}

# clean up
ssh ${REMOTE_HOST} rm -rf ${TMP_DIR}
clean_files
exit 0
