#include <stddef.h>

extern void ompc_init(int argc,char *argv[]);

void _XMP_threads_init(void) {
  ompc_init(1, NULL);
}

void _XMP_threads_finalize(void) {
  return;
}
