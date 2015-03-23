SUBDIRS = parallel kernels loop data update host_data etc

.PHONY: clean all run $(SUBDIRS)

all: $(SUBDIRS)
$(SUBDIRS):
	$(MAKE) -C $@
run:
	@for subdir in $(SUBDIRS); do $(MAKE) -C $$subdir run ; done
clean:
	@for subdir in $(SUBDIRS); do $(MAKE) -C $$subdir clean ; done
