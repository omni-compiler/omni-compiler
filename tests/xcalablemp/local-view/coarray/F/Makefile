DIRS= test_get test_put test_scope test_alloc test_fe test_cfold
#DIRS=test_hybrid

.PHONY: clean all run go

go: all

all:
	cd test_get; $(MAKE) all
	cd test_put; $(MAKE) all
	cd test_scope; $(MAKE) all
	cd test_alloc; $(MAKE) all
	cd test_fe; $(MAKE) all
	cd test_cfold; $(MAKE) all

run:
	cd test_get; $(MAKE) -j1 run
	cd test_put; $(MAKE) -j1 run
	cd test_scope; $(MAKE) -j1 run
	cd test_alloc; $(MAKE) -j1 run
	cd test_fe; $(MAKE) -j1 run
	cd test_cfold; $(MAKE) -j1 run

rerun:
	cd test_get; $(MAKE) -j1 rerun
	cd test_put; $(MAKE) -j1 rerun
	cd test_scope; $(MAKE) -j1 rerun
	cd test_alloc; $(MAKE) -j1 rerun
	cd test_fe; $(MAKE) -j1 rerun
	cd test_cfold; $(MAKE) -j1 rerun

submit:
	cd test_get; $(MAKE) submit
	cd test_put; $(MAKE) submit
	cd test_scope; $(MAKE) submit
	cd test_alloc; $(MAKE) submit
	cd test_fe; $(MAKE) submit
	cd test_cfold; $(MAKE) submit

clean:
	cd test_get; $(MAKE) clean
	cd test_put; $(MAKE) clean
	cd test_scope; $(MAKE) clean
	cd test_alloc; $(MAKE) clean
	cd test_fe; $(MAKE) clean
	cd test_cfold; $(MAKE) clean


