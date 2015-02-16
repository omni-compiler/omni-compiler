#!/bin/sh

# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $

echo aclocal
aclocal --force || exit 1
echo libtoolize
libtoolize -f -c || exit 1
echo autoheader
autoheader -f || exit 1
echo autoconf
autoconf -f || exit 1

for f in `find . -name Makefile.am -not -path \*/dist/\*`
do
    m=`basename $f .am`
    mm=`dirname $f`/$m
    echo automake $mm
#    to keep config.sub
#    automake --foreign -f -a -c $mm || exit 1
#    automake --foreign -a -c $mm || exit 1
    automake --foreign -a -c -i $mm || exit 1
done

echo delete intermediate files
rm -rf Makefile config.cache config.log config.status autom4te.cache
(cd include;
    rm -rf config.h config.h.in~ stamp-h1)
(cd C-FrontEnd;
    rm -rf src/Makefile)
(cd C-BackEnd;
    rm -rf Makefile bin/C_Back)
(cd F-FrontEnd;
    rm -rf src/Makefile)
(cd F-BackEnd;
    rm -rf Makefile bin/F_Back)
(cd Driver;
    rm -rf Makefile etc/java.conf etc/ompcc.conf \
    etc/xmpcc.conf etc/xmpf90.conf etc/ompf90.conf \
    bin/ompcc bin/ompf90 bin/xmpcc bin/xmpf90 \
    libexec/omni_traverse)

(cd libxmp;
    rm -rf src/Makefile)
(cd libompc;
    rm -rf src/Makefile)
(cd libtlog;
    rm -rf src/Makefile)
(cd XcodeML-Common;
    rm -rf src/Makefile)
(cd XcodeML-Exc-Tools;
    rm -rf src/Makefile)
(cd tests;
    rm -rf C-test/Makefile tiny/Makefile clinkpack/Makefile)

echo 'Now run ./configure'

