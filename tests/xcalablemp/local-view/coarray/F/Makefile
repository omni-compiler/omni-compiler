DIRS=test_get test_put test_fe test_cfold test_hybrid

.PHONY: clean all run go

go: all run

all:
	for d in $(DIRS); do \
	  cd $$d; make all && echo PASS || echo ERROR; \
	done

run:
	for d in $(DIRS); do \
	  cd $$d; make run && echo PASS || echo ERROR; \
	done

clean:
	for d in $(DIRS); do \
	  cd $$d; make clean && echo PASS || echo ERROR; \
	done

