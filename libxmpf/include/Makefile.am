include_HEADERS = xmp_lib.h xmp_coarray.h xmp_coarray_get.h xmp_coarray_alloc.h xmp_coarray_sync.h

xmp_coarray_get.h:  xmp_coarray_get.h.sh
	. $< $(TARGET) > $@

xmp_coarray_alloc.h:  xmp_coarray_alloc.h.sh
	. $< $(TARGET) > $@

clean-local:
	rm -f xmp_coarray_get.h
	rm -f xmp_coarray_alloc.h
