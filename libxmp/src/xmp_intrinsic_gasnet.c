#include "xmp_internal.h"

void _XMP_gasnet_atomic_define(int target_rank, _XMP_coarray_t *c, size_t offset, int value, size_t elmt_size)
{
  gasnet_put_bulk(target_rank, c->addr[target_rank] + elmt_size * offset, &value, elmt_size);
}

void _XMP_gasnet_atomic_ref(int target_rank, _XMP_coarray_t *c, size_t offset, int *value, size_t elmt_size)
{
  gasnet_get_bulk(value, target_rank, c->addr[target_rank] + elmt_size * offset, elmt_size);
}

