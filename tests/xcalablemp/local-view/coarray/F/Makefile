DIRS=./test_fe ../F_get ../F_put ../F_alloc ../F_scope ../F_cfold
DIRS_SUBMIT=../F_get ../F_put ../F_alloc ../F_scope

.PHONY: clean all run allall runall rerun submit cleanall

all:
	cd test_fe; $(MAKE) all

run:
	cd test_fe; $(MAKE) -j1 run

clean:
	cd test_fe; $(MAKE) clean



allall:
	for d in $(DIRS); do \
	  $(MAKE) -C $$d all; \
	done

runall:
	for d in $(DIRS); do \
	  $(MAKE) -C $$d -j1 run; \
	done

rerun:
	for d in $(DIRS); do \
	  $(MAKE) -C $$d -j1 rerun; \
	done

submit:
	for d in $(DIRS_SUBMIT); do \
	  $(MAKE) -C $$d submit; \
	done

cleanall:
	for d in $(DIRS); do \
	  $(MAKE) -C $$d clean; \
	done

