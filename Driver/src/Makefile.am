bin_PROGRAMS = xmp_collect_init
bin_SCRIPTS  = ../bin/ompcc ../bin/ompf90 ../bin/xmpf90 ../bin/xmpcc
sysconf_DATA = ../etc/java.conf ../etc/version ../etc/xmpcc.conf ../etc/xmpf90.conf \
               ../etc/ompcc.conf ../etc/ompf90.conf
dist_libexec_SCRIPTS = ../libexec/omni_common_lib.sh ../libexec/xmpf90_lib.sh ../libexec/xmpcc_lib.sh \
                       ../libexec/ompcc_lib.sh ../libexec/ompf90_lib.sh \
                       ../libexec/omni_traverse om_replace_pragma
xmp_collect_init_SOURCES=xmp_collect_init.c
xmp_collect_init_CPPFLAGS=-DNM="\"$(NM)\""
