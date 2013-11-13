#include <stdio.h>
#include "acc_internal.h"

void *_ACC_alloc(size_t size) {
  void *addr;

  addr = malloc(size);
  if (addr == NULL) {
    _ACC_fatal("cannot allocate memory");
  }

  return addr;
}

void _ACC_free(void *p) {
  free(p);
}

void _ACC_fatal(const char *msg) {
  fprintf(stderr, "OpenACC runtime error: %s\n", msg);
  exit(1);
}

void _ACC_unexpected_error(void) {
  _ACC_fatal("unexpected error in runtime");
}
