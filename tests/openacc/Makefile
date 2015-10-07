SUBDIRS = parallel kernels loop data declare update host_data etc atomic runtime

.PHONY: clean all run $(SUBDIRS)

all: $(SUBDIRS)
$(SUBDIRS):
	$(MAKE) -C $@
run:
	@for subdir in $(SUBDIRS); do $(MAKE) -C $$subdir run ; done
clean:
	@for subdir in $(SUBDIRS); do $(MAKE) -C $$subdir clean ; done
