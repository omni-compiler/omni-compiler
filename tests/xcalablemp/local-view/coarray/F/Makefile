DIRS=MIX FE ALLOC LIB GET PUT SCOPE CFOLD
DIRS_SUBMIT=MIX ALLOC LIB GET PUT SCOPE CFOLD

# usage: make [OPT=--debug] [all4|all6]


.PHONY: clean all run rerun submit

all:
	for d in $(DIRS); do \
	  $(MAKE) -C $$d all; \
	done

all4:
	for d in $(DIRS); do \
	  $(MAKE) -C $$d all4; \
	done

all6:
	for d in $(DIRS); do \
	  $(MAKE) -C $$d all6; \
	done

run:
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

submit4:
	for d in $(DIRS_SUBMIT); do \
	  $(MAKE) -C $$d submit4; \
	done

submit6:
	for d in $(DIRS_SUBMIT); do \
	  $(MAKE) -C $$d submit6; \
	done

clean:
	for d in $(DIRS); do \
	  $(MAKE) -C $$d clean; \
	done

clean-all:
	for d in $(DIRS); do \
	  $(MAKE) -C $$d clean-all; \
	done

