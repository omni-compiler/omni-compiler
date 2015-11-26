#include "xmp_internal.h"

/**
   Wrapper function of executing Lock.
   The second argument "offset" is a distance from beginning address of lock object.
   e.g. xmp_lock_t lockobj[a][b][c]:[*]; #pragma xmp lock(lockobj[3][2][1]:[x]) -> offset = 3*b*c + 2*c + 1
*/
static void _XMP_lock(_XMP_coarray_t* c, const unsigned int offset, const unsigned int rank)
{
#ifdef _XMP_GASNET
  _xmp_gasnet_lock(c, offset, rank);
#else
  _XMP_fatal("Cannt use lock Function");
#endif
}

/**
   Wrapper function of executing Lock using local coarray (#pragma xmp lock(lockobj[offset]))
*/
void _XMP_lock_0(_XMP_coarray_t* c, const unsigned int offset, void* lock_obj)
{
  _XMP_lock(c, offset, _XMP_world_rank);
}

/**
   Wrapper function of executing Lock using 1-dim coarray (#pragma xmp lock(lockobj[offset]:[e0]))
*/
void _XMP_lock_1(_XMP_coarray_t* c, const unsigned int offset, const unsigned int e0)
{
  unsigned int rank = e0-1;
  _XMP_lock(c, offset, rank);
}

/**
   Wrapper function of executing Lock using 2-dim coarray (#pragma xmp lock(lockobj[offset]:[e0][e1]))
*/
void _XMP_lock_2(_XMP_coarray_t* c, const unsigned int offset, const unsigned int e0, const unsigned int e1)
{
  unsigned int rank = (e0-1) + c->distance_of_image_elmts[1] * (e1-1);
  _XMP_lock(c, offset, rank);
}

/**
   Wrapper function of executing Lock using 3-dim coarray (#pragma xmp lock(lockobj[offset]:[e0][e1][e2]))
*/
void _XMP_lock_3(_XMP_coarray_t* c, const unsigned int offset, const unsigned int e0, const unsigned int e1,
		 const unsigned int e2)
{
  unsigned int rank = (e0-1) + c->distance_of_image_elmts[1] * (e1-1) + c->distance_of_image_elmts[2] * (e2-1);
  _XMP_lock(c, offset, rank);
}

/**
   Wrapper function of executing Lock using 4-dim coarray (#pragma xmp lock(lockobj[offset]:[e0][e1][e2][e3]))
*/
void _XMP_lock_4(_XMP_coarray_t* c, const unsigned int offset, const unsigned int e0, const unsigned int e1,
		 const unsigned int e2, const unsigned int e3)
{
  unsigned int rank = (e0-1) + c->distance_of_image_elmts[1] * (e1-1) + c->distance_of_image_elmts[2] * (e2-1) +
    c->distance_of_image_elmts[3] * (e3-1);
  _XMP_lock(c, offset, rank);
}

/**
   Wrapper function of executing Lock using 5-dim coarray (#pragma xmp lock(lockobj[offset]:[e0][e1][e2][e3][e4]))
*/
void _XMP_lock_5(_XMP_coarray_t* c, const unsigned int offset, const unsigned int e0, const unsigned int e1,
		 const unsigned int e2, const unsigned int e3, const unsigned int e4)
{
  unsigned int rank = (e0-1) + c->distance_of_image_elmts[1] * (e1-1) + c->distance_of_image_elmts[2] * (e2-1) +
    c->distance_of_image_elmts[3] * (e3-1) + c->distance_of_image_elmts[4] * (e4-1);
  _XMP_lock(c, offset, rank);
}

/**
   Wrapper function of executing Lock using 6-dim coarray (#pragma xmp lock(lockobj[offset]:[e0][e1][e2][e3][e4][e5]))
*/
void _XMP_lock_6(_XMP_coarray_t* c, const unsigned int offset, const unsigned int e0, const unsigned int e1,
		 const unsigned int e2, const unsigned int e3, const unsigned int e4, const unsigned int e5)
{
  unsigned int rank = (e0-1) + c->distance_of_image_elmts[1] * (e1-1) + c->distance_of_image_elmts[2] * (e2-1) +
    c->distance_of_image_elmts[3] * (e3-1) + c->distance_of_image_elmts[4] * (e4-1) + c->distance_of_image_elmts[5] * (e5-1);
  _XMP_lock(c, offset, rank);
}

/**
   Wrapper function of executing Lock using 7-dim coarray (#pragma xmp lock(lockobj[offset]:[e0][e1][e2][e3][e4][e5][e6]))
*/
void _XMP_lock_7(_XMP_coarray_t* c, const unsigned int offset, const unsigned int e0, const unsigned int e1,
		 const unsigned int e2, const unsigned int e3, const unsigned int e4, const unsigned int e5,
		 const unsigned int e6)
{
  unsigned int rank = (e0-1) + c->distance_of_image_elmts[1] * (e1-1) + c->distance_of_image_elmts[2] * (e2-1) +
    c->distance_of_image_elmts[3] * (e3-1) + c->distance_of_image_elmts[4] * (e4-1) + c->distance_of_image_elmts[5] * (e5-1) +
    c->distance_of_image_elmts[6] * (e6-1);
  _XMP_lock(c, offset, rank);
}

/**
 Wrapper function of executing Unlock.
 The second argument "offset" is a distance from beginning address of lock object.
e.g. xmp_lock_t lockobj[10][10]; #pragma xmp unlock(lockobj[3][2]:[x]) -> offset = 32
*/
void _XMP_unlock(_XMP_coarray_t* c, const unsigned int offset, const unsigned int rank)
{
#ifdef _XMP_GASNET
  _xmp_gasnet_unlock(c, offset*sizeof(xmp_lock_t), rank);
#else
  _XMP_fatal("Cannt use unlock Function");
#endif
}

/**
   Wrapper function of executing Lock using local coarray (#pragma xmp lock(lockobj[offset]))
*/
void _XMP_unlock_0(_XMP_coarray_t* c, const unsigned int offset, void* lock_obj)
{
  _XMP_unlock(c, offset, _XMP_world_rank);
}

/**
   Wrapper function of executing Lock using 1-dim coarray (#pragma xmp lock(lockobj[offset]:[e0]))
*/
void _XMP_unlock_1(_XMP_coarray_t* c, const unsigned int offset, const unsigned int e0)
{
  unsigned int rank = e0-1;
  _XMP_unlock(c, offset, rank);
}

/**
   Wrapper function of executing Lock using 2-dim coarray (#pragma xmp lock(lockobj[offset]:[e0][e1]))
*/
void _XMP_unlock_2(_XMP_coarray_t* c, const unsigned int offset, const unsigned int e0, const unsigned int e1)
{
  unsigned int rank = (e0-1) + c->distance_of_image_elmts[1] * (e1-1);
  _XMP_unlock(c, offset, rank);
}

/**
   Wrapper function of executing Lock using 3-dim coarray (#pragma xmp lock(lockobj[offset]:[e0][e1][e2]))
*/
void _XMP_unlock_3(_XMP_coarray_t* c, const unsigned int offset, const unsigned int e0, const unsigned int e1,
		   const unsigned int e2)
{
  unsigned int rank = (e0-1) + c->distance_of_image_elmts[1] * (e1-1) + c->distance_of_image_elmts[2] * (e2-1);
  _XMP_unlock(c, offset, rank);
}

/**
   Wrapper function of executing Lock using 4-dim coarray (#pragma xmp lock(lockobj[offset]:[e0][e1][e2][e3]))
*/
void _XMP_unlock_4(_XMP_coarray_t* c, const unsigned int offset, const unsigned int e0, const unsigned int e1,
		   const unsigned int e2, const unsigned int e3)
{
  unsigned int rank = (e0-1) + c->distance_of_image_elmts[1] * (e1-1) + c->distance_of_image_elmts[2] * (e2-1) +
    c->distance_of_image_elmts[3] * (e3-1);
  _XMP_unlock(c, offset, rank);
}
/**
   Wrapper function of executing Lock using 5-dim coarray (#pragma xmp lock(lockobj[offset]:[e0][e1][e2][e3][e4]))
*/
void _XMP_unlock_5(_XMP_coarray_t* c, const unsigned int offset, const unsigned int e0, const unsigned int e1,
		   const unsigned int e2, const unsigned int e3, const unsigned int e4)
{
  unsigned int rank = (e0-1) + c->distance_of_image_elmts[1] * (e1-1) + c->distance_of_image_elmts[2] * (e2-1) +
    c->distance_of_image_elmts[3] * (e3-1) + c->distance_of_image_elmts[4] * (e4-1);
  _XMP_unlock(c, offset, rank);
}

/**
   Wrapper function of executing Lock using 6-dim coarray (#pragma xmp lock(lockobj[offset]:[e0][e1][e2][e3][e4][e5]))
*/
void _XMP_unlock_6(_XMP_coarray_t* c, const unsigned int offset, const unsigned int e0, const unsigned int e1,
		   const unsigned int e2, const unsigned int e3, const unsigned int e4, const unsigned int e5)
{
  unsigned int rank = (e0-1) + c->distance_of_image_elmts[1] * (e1-1) + c->distance_of_image_elmts[2] * (e2-1) +
    c->distance_of_image_elmts[3] * (e3-1) + c->distance_of_image_elmts[4] * (e4-1) + c->distance_of_image_elmts[5] * (e5-1);
  _XMP_unlock(c, offset, rank);
}

/**
   Wrapper function of executing Lock using 7-dim coarray (#pragma xmp lock(lockobj[offset]:[e0][e1][e2][e3][e4][e5][e6]))
*/
void _XMP_unlock_7(_XMP_coarray_t* c, const unsigned int offset, const unsigned int e0, const unsigned int e1,
		   const unsigned int e2, const unsigned int e3, const unsigned int e4, const unsigned int e5,
		   const unsigned int e6)
{
  unsigned int rank = (e0-1) + c->distance_of_image_elmts[1] * (e1-1) + c->distance_of_image_elmts[2] * (e2-1) +
    c->distance_of_image_elmts[3] * (e3-1) + c->distance_of_image_elmts[4] * (e4-1) + c->distance_of_image_elmts[5] * (e5-1) +
    c->distance_of_image_elmts[6] * (e6-1);
  _XMP_unlock(c, offset, rank);
}

/**
   Wrapper function of initializing Lock object
*/
static void _xmp_lock_initialize(void* addr, const unsigned int number_of_elements)
{
#ifdef _XMP_GASNET
  _xmp_gasnet_lock_initialize(addr, number_of_elements);
#else
  _XMP_fatal("Cannt use lock Function");
#endif
}

/**
   Wrapper function of initializing 1-dim array Lock object (e.g. xmp_lock_t A[e0];)
*/
void _XMP_lock_initialize_1(void *addr, const unsigned int e0)
{
  _xmp_lock_initialize(addr, e0);
}

/**
   Wrapper function of initializing 2-dim array Lock object (e.g. xmp_lock_t A[e0][e1];)
*/
void _XMP_lock_initialize_2(void *addr, const unsigned int e0, const unsigned int e1)
{
  _xmp_lock_initialize(addr, e0*e1);
}

/**
   Wrapper function of initializing 3-dim array Lock object (e.g. xmp_lock_t A[e0][e1][e2];)
*/
void _XMP_lock_initialize_3(void *addr, const unsigned int e0, const unsigned int e1, const unsigned int e2)
{
  _xmp_lock_initialize(addr, e0*e1*e2);
}

/**
   Wrapper function of initializing 4-dim array Lock object (e.g. xmp_lock_t A[e0][e1][e2][e3];)
*/
void _XMP_lock_initialize_4(void *addr, const unsigned int e0, const unsigned int e1, const unsigned int e2,
			    const unsigned int e3)
{
  _xmp_lock_initialize(addr, e0*e1*e2*e3);
}

/**
   Wrapper function of initializing 5-dim array Lock object (e.g. xmp_lock_t A[e0][e1][e2][e3][e4];)
*/
void _XMP_lock_initialize_5(void *addr, const unsigned int e0, const unsigned int e1, const unsigned int e2,
			    const unsigned int e3, const unsigned int e4)
{
  _xmp_lock_initialize(addr, e0*e1*e2*e3*e4);
}

/**
   Wrapper function of initializing 6-dim array Lock object (e.g. xmp_lock_t A[e0][e1][e2][e3][e4][e5];)
*/
void _XMP_lock_initialize_6(void *addr, const unsigned int e0, const unsigned int e1, const unsigned int e2,
			    const unsigned int e3, const unsigned int e4, const unsigned int e5)
{
  _xmp_lock_initialize(addr, e0*e1*e2*e3*e4*e5);
}

/**
   Wrapper function of initializing 7-dim array Lock object (e.g. xmp_lock_t A[e0][e1][e2][e3][e4][e5][e6];)
 */
void _XMP_lock_initialize_7(void *addr, const unsigned int e0, const unsigned int e1, const unsigned int e2,
			    const unsigned int e3, const unsigned int e4, const unsigned int e5, const unsigned int e6)
{
  _xmp_lock_initialize(addr, e0*e1*e2*e3*e4*e5*e6);
}
