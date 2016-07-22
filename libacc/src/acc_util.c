#include <stdio.h>
#include "acc_internal.h"
#include <signal.h>
#include <limits.h>

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
  raise(SIGABRT);
  exit(1);
}

void _ACC_unexpected_error(void) {
  _ACC_fatal("unexpected error in runtime");
}

static inline
int ceilll(long long a, long long b)
{
  return (a-1)/b + 1;
}

int _ACC_adjust_num_gangs(long long num_gangs, int limit)
{
  if(_ACC_num_gangs_limit == 0){
    limit = INT_MAX;
  }

  if(_ACC_num_gangs_limit > 0){
    limit = _ACC_num_gangs_limit;
  }

  if(num_gangs > limit){
    return ceilll(num_gangs, ceilll(num_gangs, limit));
  }else{
    return (int)num_gangs;
  }
}
