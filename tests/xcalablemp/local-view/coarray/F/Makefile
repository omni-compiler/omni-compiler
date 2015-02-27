#DIRS=test_get test_put test_fe test_cfold test_hybrid
DIRS=test_get test_put test_fe test_cfold

.PHONY: clean all run go

go: all

all:
	for d in $(DIRS); do \
	  (cd $$d; $(MAKE) all) && echo PASS || echo "ERROR!!"; \
	done

run:
	for d in $(DIRS); do \
	  (cd $$d; $(MAKE) -j1 run) && echo PASS || echo "ERROR!!"; \
	done

submit:
	for d in $(DIRS); do \
	  (cd $$d; $(MAKE) submit) && echo PASS || echo "ERROR!!"; \
	done

clean:
	for d in $(DIRS); do \
	  (cd $$d; $(MAKE) clean) && echo PASS || echo "ERROR!!"; \
	done

