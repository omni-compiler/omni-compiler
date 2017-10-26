.PHONY: all install clean

LIBDIR    = /home/mnakao/work/xmp-trunk/lib
MKDIR_P   = install -m 755 -d
INSTALL   = install

all: install

install: xmp.py
	$(MKDIR_P) $(DESTDIR)$(LIBDIR)
	$(INSTALL) $^ $(DESTDIR)$(LIBDIR)

clean:
	rm -rf *~ *.pyc

