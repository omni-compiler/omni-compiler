/*
 *   COARRAY ATOMIC define/ref, add/and, caf (compare and swap)
 *
 */

#include "xmpf_internal.h"

/* cannot be included because of many conflicts with xmp_internal.h
 */
//#include "xmp_func_decl.h"
extern void _XMP_atomic_define_0(void *, size_t, int, void*, size_t, size_t);
extern void _XMP_atomic_define_1(void *, size_t, int, int, void*, size_t, size_t);

extern void _XMP_atomic_ref_0(void *, size_t, int*, void *, size_t, size_t);
extern void _XMP_atomic_ref_1(void *, size_t, int, int*, void *, size_t, size_t);


static void _atomic_define_self_core(void *descPtr, int *atomAddr, int *srcAddr);
static void _atomic_define_remote_core(void *descPtr, int coindex, int *moldAddr,
                                       int *srcAddr);

static void _atomic_ref_self_core(void *descPtr, int *atomAddr, int *dstAddr);
static void _atomic_ref_remote_core(void *descPtr, int coindex, int *moldAddr,
                                    int *dstAddr);


/*-----------------------------------------------------------------------*\
 *   atomic_define
 *     generic subroutine xmpf_atomic_define_generic
 *       declared in ../include/xmp_coarray_atomic.h
\*-----------------------------------------------------------------------*/

/*
 *  xmpf_atomic_define_self_{i4,l4}_
 */
void xmpf_atomic_define_self_i4_(void **descPtr, int *atom, int *src)
{
  _atomic_define_self_core(*descPtr, atom, src);
}

void xmpf_atomic_define_self_l4_(void **descPtr, int *atom, int *src)
{
  _atomic_define_self_core(*descPtr, atom, src);
}

static void _atomic_define_self_core(void *descPtr, int *atomAddr, int *srcAddr)
{
  void *atomDesc, *srcDesc = NULL;
  size_t atomOffset, srcOffset = 0;
  char *srcOrgAddr = NULL;
  char *srcName;

  /* get the descriptor of src (if any)
   */
  srcDesc = _XMPF_get_coarrayDescFromAddr((char*)srcAddr, &srcOrgAddr,
                                          &srcOffset, &srcName);

  /* get the descriptor of atom
   */
  atomDesc = _XMPF_get_coarrayDesc(descPtr);
  atomOffset = _XMPF_get_coarrayOffset(descPtr, (char*)atomAddr);

  /* action
   */
  _XMP_atomic_define_0(atomDesc, atomOffset/sizeof(int),
                       *srcAddr, srcDesc, srcOffset/sizeof(int), sizeof(int));
}


/*
 *  xmpf_atomic_define_remote_{i4,l4}_
 */
void xmpf_atomic_define_remote_i4_(void **descPtr, int *coindex, int *mold, int *src)
{
  _atomic_define_remote_core(*descPtr, *coindex, mold, src);
}

void xmpf_atomic_define_remote_l4_(void **descPtr, int *coindex, int *mold, int *src)
{
  _atomic_define_remote_core(*descPtr, *coindex, mold, src);
}

static void _atomic_define_remote_core(void *descPtr, int coindex, int *moldAddr,
                                       int *srcAddr)
{
  void *atomDesc, *srcDesc = NULL;
  size_t atomOffset, srcOffset = 0;
  char *srcOrgAddr = NULL;
  char *srcName;

  /* get the descriptor of src (if any)
   */
  srcDesc = _XMPF_get_coarrayDescFromAddr((char*)srcAddr, &srcOrgAddr,
                                          &srcOffset, &srcName);

  /* get the descriptor of atom (remote coarray)
   */
  atomDesc = _XMPF_get_coarrayDesc(descPtr);
  atomOffset = _XMPF_get_coarrayOffset(descPtr, (char*)moldAddr);
  int image = _XMPF_get_initial_image_withDescPtr(coindex, descPtr);

  /* action
   */
  _XMP_atomic_define_1(atomDesc, atomOffset/sizeof(int), image,
                       *srcAddr, srcDesc, srcOffset/sizeof(int), sizeof(int));
}


/*-----------------------------------------------------------------------*\
 *   atomic_ref
 *     generic subroutine xmpf_atomic_ref_generic
 *       declared in ../include/xmp_coarray_atomic.h
\*-----------------------------------------------------------------------*/

/*
 *  xmpf_atomic_ref_self_{i4,l4}_
 */
void xmpf_atomic_ref_self_i4_(void **descPtr, int *atom, int *dst)
{
  _atomic_ref_self_core(*descPtr, atom, dst);
}

void xmpf_atomic_ref_self_l4_(void **descPtr, int *atom, int *dst)
{
  _atomic_ref_self_core(*descPtr, atom, dst);
}

static void _atomic_ref_self_core(void *descPtr, int *atomAddr, int *dstAddr)
{
  void *atomDesc, *dstDesc = NULL;
  size_t atomOffset, dstOffset = 0;
  char *dstOrgAddr = NULL;
  char *dstName;

  /* get the descriptor of dst (if any)
   */
  dstDesc = _XMPF_get_coarrayDescFromAddr((char*)dstAddr, &dstOrgAddr,
                                          &dstOffset, &dstName);

  /* get the descriptor of atom
   */
  atomDesc = _XMPF_get_coarrayDesc(descPtr);
  atomOffset = _XMPF_get_coarrayOffset(descPtr, (char*)atomAddr);

  /* action
   */
  _XMP_atomic_ref_0(atomDesc, atomOffset/sizeof(int),
                    dstAddr, dstDesc, dstOffset/sizeof(int), sizeof(int));
}


/*
 *  xmpf_atomic_ref_remote_{i4,l4}_
 */
void xmpf_atomic_ref_remote_i4_(void **descPtr, int *coindex, int *mold, int *dst)
{
  _atomic_ref_remote_core(*descPtr, *coindex, mold, dst);
}

void xmpf_atomic_ref_remote_l4_(void **descPtr, int *coindex, int *mold, int *dst)
{
  _atomic_ref_remote_core(*descPtr, *coindex, mold, dst);
}

static void _atomic_ref_remote_core(void *descPtr, int coindex, int *moldAddr,
                                    int *dstAddr)
{
  void *atomDesc, *dstDesc = NULL;
  size_t atomOffset, dstOffset = 0;
  char *dstOrgAddr = NULL;
  char *dstName;

  /* get the descriptor of dst (if any)
   */
  dstDesc = _XMPF_get_coarrayDescFromAddr((char*)dstAddr, &dstOrgAddr,
                                          &dstOffset, &dstName);

  /* get the descriptor of atom (remote coarray)
   */
  atomDesc = _XMPF_get_coarrayDesc(descPtr);
  atomOffset = _XMPF_get_coarrayOffset(descPtr, (char*)moldAddr);
  int image = _XMPF_get_initial_image_withDescPtr(coindex, descPtr);

  /* action
   */
  _XMP_atomic_ref_1(atomDesc, atomOffset/sizeof(int), image,
                    dstAddr, dstDesc, dstOffset/sizeof(int), sizeof(int));
}


