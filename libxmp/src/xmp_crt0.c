#include "xmp_internal.h"

extern int _XCALABLEMP_main(int argc, char *argv[]);

int main(int argc, char *argv[]) {
  _XCALABLEMP_init_world(&argc, &argv);

  _XCALABLEMP_barrier_WORLD();
  int ret = _XCALABLEMP_main(argc, argv);
  _XCALABLEMP_barrier_WORLD();

  return _XCALABLEMP_finalize_world(ret);
}
