#!/bin/sh

# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $

echo aclocal
aclocal --force || exit 1
echo autoheader
autoheader -f || exit 1
echo autoconf
autoconf -f || exit 1

for f in `find . -name Makefile.am -not -path \*/dist/\*`
do
    m=`basename $f .am`
    mm=`dirname $f`/$m
    echo automake $mm
    automake --foreign -f -a -c $mm || exit 1
done

echo delete intermediate files
rm -rf Makefile config.cache config.log config.status autom4te.cache
(cd include;
    rm -rf config.h config.h.in~ stamp-h1)
(cd C-FrontEnd;
    rm -rf src/Makefile)
(cd C-BackEnd;
    rm -rf Makefile ant.properties bin/C_Back)
(cd F-FrontEnd;
    rm -rf src/Makefile)
(cd F-BackEnd;
    rm -rf Makefile ant.properties bin/F_Back)
(cd Driver;
    rm -rf Makefile etc/omc.conf etc/omf.conf \
    etc/java.conf \
    etc/*.openmp \
    bin/*.openmp \
    bin/oml2x bin/omx2x bin/omx2l \
    bin/ompp bin/omnative bin/omlinker \
    bin/omc2c bin/ompcc bin/omc2x bin/omcx2x bin/omx2c \
    bin/omcpp bin/omcnative bin/omclinker \
    bin/omf2f bin/ompf90 bin/omf2x bin/omfx2x bin/omx2f \
    bin/omfpp bin/omfnative bin/omflinker \
    bin/xmpcc bin/*.xmp etc/*.xmp \
    bin/xmpcc-threads bin/*.xmp_threads etc/*.xmp_threads)

(cd libxmp;
    rm -rf src/Makefile)
(cd libompc;
    rm -rf src/Makefile)
(cd libtlog;
    rm -rf src/Makefile)
(cd XcodeML-Common;
    rm -rf src/Makefile src/ant.properties)
(cd XcodeML-Exc-Tools;
    rm -rf src/Makefile src/ant.properties)
(cd tests;
    rm -rf C-test/Makefile tiny/Makefile clinkpack/Makefile)

echo 'Now run ./configure'


