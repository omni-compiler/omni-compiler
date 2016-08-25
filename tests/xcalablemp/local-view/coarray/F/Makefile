DIRS=STRUCT MIX ALLOC LIB GET PUT SCOPE CFOLD
DIRS_SUBMIT=STRUCT MIX ALLOC LIB GET PUT SCOPE CFOLD

# To make all executable files, type:
#   make [OPT=--debug] [all|all4|all6|all7|all7g]

# Ver.3 (default) uses common cray pointers to point the coarray objects.
# Ver.4 makes coarray objects common without using cray pointers.
#   Restriction: only for FJRDMA and MPI3.
# Ver.6 initializes procedure-local coarray objects at the first entrance
#   of the procedure. Restriction: only for FJRDMA and MPI3 and only for
#   execution with whole images.
# Ver.7 (for FJRDMA and MPI3) and Ver.7g (for GASNet) use the ENTRY 
#   statement and initializes procedure-local coarray objects before 
#   execution of the program.
# For more infos., see the blue note and comment in XMPtransCoarrayRun.java


.PHONY: clean all run rerun submit showlog cleanlog

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

all7:
	for d in $(DIRS); do \
	  $(MAKE) -C $$d all7 || exit 1; \
	done

all7g:
	for d in $(DIRS); do \
	  $(MAKE) -C $$d all7g || exit 1; \
	done

run:
	for d in $(DIRS); do \
	  $(MAKE) -C $$d -j1 run || exit 1; \
	done

run4:
	for d in $(DIRS); do \
	  $(MAKE) -C $$d -j1 run4 || exit 1; \
	done

run6:
	for d in $(DIRS); do \
	  $(MAKE) -C $$d -j1 run6 || exit 1; \
	done

run7:
	for d in $(DIRS); do \
	  $(MAKE) -C $$d -j1 run7 || exit 1; \
	done

run7g:
	for d in $(DIRS); do \
	  $(MAKE) -C $$d -j1 run7g || exit 1; \
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

submit7:
	for d in $(DIRS_SUBMIT); do \
	  $(MAKE) -C $$d submit7 || exit 1; \
	done

submit7g:
	for d in $(DIRS_SUBMIT); do \
	  $(MAKE) -C $$d submit7g || exit 1; \
	done

showlog:
	for d in $(DIRS_SUBMIT); do \
	  $(MAKE) -C $$d showlog || exit 1; \
	done

cleanlog:
	for d in $(DIRS); do \
	  $(MAKE) -C $$d cleanlog || exit 1; \
	done

clean: cleanlog
	for d in $(DIRS); do \
	  $(MAKE) -C $$d clean || exit 1; \
	done
