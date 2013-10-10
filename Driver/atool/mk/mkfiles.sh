#!/bin/sh
# $Id: mkfiles.sh 86 2012-07-30 05:33:07Z m-hirano $

TARGET=./.files
rm -rf ${TARGET}
echo 'INPUT = \' >> ${TARGET}
find . -type f -name '*.c' -o -name '*.cpp' -o -name '*.h' | \
    awk '{ printf "%s \\\n", $1 }' | \
    grep -v 'check/' >> ${TARGET}
echo >> ${TARGET}
