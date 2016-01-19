DIRS=FE ALLOC LIB GET PUT SCOPE CFOLD
DIRS_SUBMIT=ALLOC LIB GET PUT SCOPE CFOLD

.PHONY: clean all run rerun submit

all:
	for d in $(DIRS); do \
	  $(MAKE) -C $$d all; \
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

clean:
	for d in $(DIRS); do \
	  $(MAKE) -C $$d clean; \
	done

