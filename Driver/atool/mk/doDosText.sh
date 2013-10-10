#!/bin/sh
# $Id: doDosText.sh 86 2012-07-30 05:33:07Z m-hirano $

l=`find . -type f \( \
    -name '*.h' -o \
    -name '*.c' -o \
    -name '*.cpp' -o \
    -name 'Makefile.in' \)`

for i in ${l}; do
    nkf -c ${i} > ${i}.tmp
    rm -f ${i}
    mv -f ${i}.tmp ${i}
done

exit 0


