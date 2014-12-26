# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $
bin_PROGRAMS=omdriver xmp_collect_init om_replace_pragma
bin_SCRIPTS= ../bin/ompp ../bin/oml2x ../bin/omx2x ../bin/omx2l ../bin/omnative ../bin/omlinker \
	../bin/omc2c ../bin/ompcc ../bin/omc2x ../bin/omcx2x ../bin/omx2c ../bin/omcpp \
	../bin/omcnative ../bin/omclinker ../bin/omf2f ../bin/ompf90 ../bin/omf2x ../bin/omfx2x \
        ../bin/omx2f ../bin/omfpp ../bin/omfnative ../bin/omflinker ../bin/xmpcc ../bin/xmpf90 \
	../bin/omtranslate ../bin/omni_collect_init

sysconf_DATA=../etc/omc.conf ../etc/omf.conf ../etc/java.conf ../etc/pp.conf.openmp ../etc/native.conf.openmp \
        ../etc/linker.conf.openmp ../etc/l2x.conf.openmp ../etc/x2x.conf.openmp ../etc/pp.conf.xmp \
	../etc/native.conf.xmp ../etc/linker.conf.xmp ../etc/l2x.conf.xmp ../etc/x2x.conf.xmp ../etc/pp.conf.xmpf \
	../etc/native.conf.xmpf ../etc/linker.conf.xmpf ../etc/l2x.conf.xmpf ../etc/x2x.conf.xmpf \
	../etc/pp.conf.acc ../etc/native.conf.acc ../etc/linker.conf.acc ../etc/l2x.conf.acc ../etc/x2x.conf.acc \
	../etc/setup.conf ../etc/version

omdriver_SOURCES=omdriver.c
xmp_collect_init_SOURCES=xmp_collect_init.c
xmp_collect_init_CPPFLAGS=-DNM="\"$(NM)\""

omdriver.c: omdriver.h

clean-local:
	rm -rf ../doc

html-local:
	if test ! -e "../doc"; then mkdir ../doc; fi
	doxygen >/dev/null
