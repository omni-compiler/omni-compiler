DIRS= test_get test_put test_scope test_alloc test_fe test_cfold
#DIRS=test_hybrid

.PHONY: clean all run go

go: all

all:
	for d in $(DIRS); do \
	  cd $$d; $(MAKE) all || { echo "ERROR!!"; exit 1; } \
	done

run:
	for d in $(DIRS); do \
	  cd $$d; $(MAKE) -j1 run || { echo "ERROR!!"; exit 1; } \
	done

rerun:
	for d in $(DIRS); do \
	  cd $$d; $(MAKE) -j1 rerun || { echo "ERROR!!"; exit 1; } \
	done

submit:
	for d in $(DIRS); do \
	  cd $$d; $(MAKE) submit || { echo "ERROR!!"; exit 1; } \
	done

clean:
	for d in $(DIRS); do \
	  cd $$d; $(MAKE) clean || { echo "ERROR!!"; exit 1; } \
	done

