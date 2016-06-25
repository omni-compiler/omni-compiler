bin_SCRIPTS  = ../bin/ompcc ../bin/ompf90 ../bin/xmpf90 ../bin/xmpcc
sysconf_DATA = ../etc/version ../etc/xmpcc.conf ../etc/xmpf90.conf \
               ../etc/ompcc.conf ../etc/ompf90.conf
dist_libexec_SCRIPTS = ../libexec/omni_common_lib.sh ../libexec/xmpf90_lib.sh ../libexec/xmpcc_lib.sh \
                       ../libexec/ompcc_lib.sh ../libexec/ompf90_lib.sh ./om_replace_pragma ../libexec/omni_traverse.sh ../libexec/omni_pzcc.sh

om_replace_pragma: om_replace_pragma.c
	$(CC) $^ -o $@

clean:
	rm -f om_replace_pragma
