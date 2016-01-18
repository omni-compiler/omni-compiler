include_HEADERS = xmp_lib.h xmp_coarray.h xmp_coarray_get.h xmp_coarray_alloc.h xmp_coarray_reduction.h

if SUPERUX
  COARRAY_SYNC_H = xmp_coarray_sync_sxace.h
else
  COARRAY_SYNC_H = xmp_coarray_sync.h
endif

install:
	mkdir -p $(includedir)
	$(INSTALL) -m 644 $(COARRAY_SYNC_H) $(includedir)/xmp_coarray_sync.h
	$(INSTALL) -m 644 $(include_HEADERS) $(includedir)/

xmp_coarray_get.h:  xmp_coarray_get.h.sh
	. $< $(TARGET) > $@

xmp_coarray_alloc.h:  xmp_coarray_alloc.h.sh
	. $< $(TARGET) > $@

xmp_coarray_reduction.h:  xmp_coarray_reduction.h.sh
	. $< $(TARGET) > $@

clean-local:
	rm -f xmp_coarray_get.h
	rm -f xmp_coarray_alloc.h
	rm -f xmp_coarray_reduction.h
