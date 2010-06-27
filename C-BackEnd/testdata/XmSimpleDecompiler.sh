#!/bin/sh

dir_abspath()
{
	pushd . > /dev/null
	cd $1
	if [ $? != 0 ]; then
		echo file not found : $1
	fi
	echo $PWD
	popd > /dev/null
}

BINDIR=`dirname $0`

if [ -z "${XMC}" ]; then
    XMC=`dir_abspath $BINDIR/..`
fi

_JAVA_OPT=-Xmx1024M

if [ -z "${XMC}" ]; then
    XMC=`dir_abspath $BINDIR`
fi

java ${_JAVA_OPT} -cp ${XMC}/classes:${XMC}/binding/classes:${XMC}/testclasses:${XMC}/../XcodeML-Common/classes xcodeml.util.XmSimpleDecompiler $@

