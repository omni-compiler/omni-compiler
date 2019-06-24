#!/bin/bash

function omni_pzcc()
{
    local FILE_PZC=
    local FILE_PZ=
    local TMP_DIR=.

    while getopts o:t:d:h OPT
    do
	case $OPT in
	    o) FILE_PZ=$OPTARG
	       ;;
	    t) TARGET_ARCH=$OPTARG
	       ;;
	    d) TMP_DIR=$OPTARG
	       ;;
	    h | \?) echo "Usage: [-o output] [-t target_arch] [-d Dir] [-h] input ..." 1>&2;
		return 1
		;;
	esac
    done

    shift $((OPTIND - 1))

    # check input
    if [ $# -eq 0 ]; then
	echo "no input file"
	return 2
    fi
    FILE_PZC=${@}

    # check output
    if [ "${FILE_PZ}" = "" ]; then
	echo "no output file"
	return 3
    fi

    # check env var "$OMNI_PZCL_PREFIX"
    if [ -z "$OMNI_PZCL_PREFIX" ]; then
	echo "OMNI_PZCL_PREFIX is not set"
	return 4
    fi

    #    echo tmp_dir = ${TMP_DIR}
    #    echo input = ${FILE_PZC}
    #    echo output = ${FILE_PZ}

    cat <<EOF > ${TMP_DIR}/Makefile_pzc
TARGET = ${FILE_PZ}
PZCSRC = ${FILE_PZC}
vpath %.pzc ${TMP_DIR}
PZC_TARGET_ARCH = sc32
OBJ_DIR = ${TMP_DIR}/__pzcc_tmp__
PZC_INC_DIR = -I$OMNI_HOME/include
include $OMNI_PZCL_PREFIX/make/default_pzcl_kernel.mk
EOF

    MAKEFLAGS= make -s -f ${TMP_DIR}/Makefile_pzc
}

callerinfo=(`caller`)
if [ "${callerinfo[1]}" = "NULL" ]; then
    tmpdir=`mktemp -d temp.XXXXXX`
    omni_pzcc -d $tmpdir $@
    rm -r $tmpdir
fi
