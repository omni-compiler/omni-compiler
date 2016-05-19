DIRS=MIX FE ALLOC LIB GET PUT SCOPE CFOLD
DIRS_SUBMIT=MIX ALLOC LIB GET PUT SCOPE CFOLD

# usage: make [OPT=--debug] [all4|all6]


.PHONY: clean all run rerun submit

all:
	for d in $(DIRS); do \
	  $(MAKE) -C $$d all || exit 1; \
	done

all4:
	for d in $(DIRS); do \
	  $(MAKE) -C $$d all4 || exit 1; \
	done

all6:
	for d in $(DIRS); do \
	  $(MAKE) -C $$d all6 || exit 1; \
	done

run:
	for d in $(DIRS); do \
	  $(MAKE) -C $$d -j1 run || exit 1; \
	done

rerun:
	for d in $(DIRS); do \
	  $(MAKE) -C $$d -j1 rerun || exit 1; \
	done

submit:
	for d in $(DIRS_SUBMIT); do \
	  $(MAKE) -C $$d submit || exit 1; \
	done

submit4:
	for d in $(DIRS_SUBMIT); do \
	  $(MAKE) -C $$d submit4 || exit 1; \
	done

submit6:
	for d in $(DIRS_SUBMIT); do \
	  $(MAKE) -C $$d submit6 || exit 1; \
	done

clean:
	for d in $(DIRS); do \
	  $(MAKE) -C $$d clean || exit 1; \
	done

clean-all:
	for d in $(DIRS); do \
	  $(MAKE) -C $$d clean-all || exit 1; \
	done

