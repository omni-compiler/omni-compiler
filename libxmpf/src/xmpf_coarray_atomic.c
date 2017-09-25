/*
 *   COARRAY ATOMIC define/ref, add/and, caf (compare and swap)
 *
 */

#include "xmpf_internal_coarray.h"

/* cannot be included because of many conflicts with xmp_internal.h
 */
//#include "xmp_func_decl.h"
extern void _XMP_atomic_define_0(void *, size_t, int, void*, size_t, size_t);
extern void _XMP_atomic_define_1(void *, size_t, int, int, void*, size_t, size_t);

extern void _XMP_atomic_ref_0(void *, size_t, int*, void *, size_t, size_t);
extern void _XMP_atomic_ref_1(void *, size_t, int, int*, void *, size_t, size_t);


static void _atomic_define_self_core(CoarrayInfo_t *cinfo, int *atomAddr, int *srcAddr);
static void _atomic_define_remote_core(CoarrayInfo_t *cinfo, int coindex,
                                       int *moldAddr, int *srcAddr);

static void _atomic_ref_self_core(CoarrayInfo_t *cinfo, int *atomAddr, int *dstAddr);
static void _atomic_ref_remote_core(CoarrayInfo_t *cinfo, int coindex,
                                    int *moldAddr, int *dstAddr);


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
  _atomic_define_self_core((CoarrayInfo_t*)(*descPtr), atom, src);
}

void xmpf_atomic_define_self_l4_(void **descPtr, int *atom, int *src)
{
  _atomic_define_self_core((CoarrayInfo_t*)(*descPtr), atom, src);
}

void _atomic_define_self_core(CoarrayInfo_t *cinfo, int *atomAddr, int *srcAddr)
{
  void *atomDesc, *srcDesc = NULL;
  size_t atomOffset, srcOffset;
  size_t srcOffset_char;
  char *srcOrgAddr_char, *atomOrgAddr_char;
  char *srcName;

  /* get the descriptor of src (if any)
   */
  srcDesc = _XMPCO_get_desc_fromLocalAddr((char*)srcAddr, &srcOrgAddr_char,
                                          &srcOffset_char, &srcName);
  srcOffset = srcDesc ? (srcAddr - (int*)srcOrgAddr_char) : 0;

  /* get the descriptor of atom
   */
  atomDesc = _XMPCO_get_descForMemoryChunk(cinfo);
  atomOrgAddr_char = _XMPCO_get_orgAddrOfMemoryChunk(cinfo);
  atomOffset = atomAddr - (int*)atomOrgAddr_char;

  /* action
   */
  _XMP_atomic_define_0(atomDesc, atomOffset,
                       *srcAddr, srcDesc, srcOffset, sizeof(int));
}


/*
 *  xmpf_atomic_define_remote_{i4,l4}_
 */
void xmpf_atomic_define_remote_i4_(void **descPtr, int *coindex, int *mold, int *src)
{
  _atomic_define_remote_core((CoarrayInfo_t*)(*descPtr), *coindex, mold, src);
}

void xmpf_atomic_define_remote_l4_(void **descPtr, int *coindex, int *mold, int *src)
{
  _atomic_define_remote_core((CoarrayInfo_t*)(*descPtr), *coindex, mold, src);
}

void _atomic_define_remote_core(CoarrayInfo_t *cinfo, int coindex,
                                int *moldAddr, int *srcAddr)
{
  void *atomDesc, *srcDesc = NULL;
  size_t moldOffset, srcOffset;
  size_t srcOffset_char;
  char *srcOrgAddr_char, *moldOrgAddr_char;
  char *srcName;

  /* get the descriptor of src (if any)
   */
  srcDesc = _XMPCO_get_desc_fromLocalAddr((char*)srcAddr, &srcOrgAddr_char,
                                          &srcOffset_char, &srcName);
  srcOffset = srcDesc ? (srcAddr - (int*)srcOrgAddr_char) : 0;

  /* get the descriptor of atom (remote coarray)
   */
  atomDesc = _XMPCO_get_descForMemoryChunk(cinfo);
  moldOrgAddr_char = _XMPCO_get_orgAddrOfMemoryChunk(cinfo);
  moldOffset = moldAddr - (int*)moldOrgAddr_char;

  int image = _XMPCO_get_initial_image_withDescPtr(coindex, cinfo);

  /* action
   */
  _XMP_atomic_define_1(atomDesc, moldOffset, image-1,
                       *srcAddr, srcDesc, srcOffset, sizeof(int));
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
  _atomic_ref_self_core((CoarrayInfo_t*)(*descPtr), atom, dst);
}

void xmpf_atomic_ref_self_l4_(void **descPtr, int *atom, int *dst)
{
  _atomic_ref_self_core((CoarrayInfo_t*)(*descPtr), atom, dst);
}

void _atomic_ref_self_core(CoarrayInfo_t *cinfo, int *atomAddr, int *dstAddr)
{
  void *atomDesc, *dstDesc = NULL;
  size_t atomOffset, dstOffset;
  size_t dstOffset_char;
  char *dstOrgAddr_char, *atomOrgAddr_char;
  char *dstName;

  /* get the descriptor of dst (if any)
   */
  dstDesc = _XMPCO_get_desc_fromLocalAddr((char*)dstAddr, &dstOrgAddr_char,
                                          &dstOffset_char, &dstName);
  dstOffset = dstDesc ? (dstAddr - (int*)dstOrgAddr_char) : 0;

  /* get the descriptor of atom
   */
  atomDesc = _XMPCO_get_descForMemoryChunk(cinfo);
  atomOrgAddr_char = _XMPCO_get_orgAddrOfMemoryChunk(cinfo);
  atomOffset = atomAddr - (int*)atomOrgAddr_char;

  /* action
   */
  _XMP_atomic_ref_0(atomDesc, atomOffset,
                    dstAddr, dstDesc, dstOffset, sizeof(int));
}


/*
 *  xmpf_atomic_ref_remote_{i4,l4}_
 */
void xmpf_atomic_ref_remote_i4_(void **descPtr, int *coindex, int *mold, int *dst)
{
  _atomic_ref_remote_core((CoarrayInfo_t*)(*descPtr), *coindex, mold, dst);
}

void xmpf_atomic_ref_remote_l4_(void **descPtr, int *coindex, int *mold, int *dst)
{
  _atomic_ref_remote_core((CoarrayInfo_t*)(*descPtr), *coindex, mold, dst);
}

void _atomic_ref_remote_core(CoarrayInfo_t *cinfo, int coindex,
                             int *moldAddr, int *dstAddr)
{
  void *atomDesc, *dstDesc = NULL;
  size_t moldOffset, dstOffset;
  size_t dstOffset_char;
  char *dstOrgAddr_char, *moldOrgAddr_char;
  char *dstName;

  /* get the descriptor of dst (if any)
   */
  dstDesc = _XMPCO_get_desc_fromLocalAddr((char*)dstAddr, &dstOrgAddr_char,
                                          &dstOffset_char, &dstName);
  dstOffset = dstDesc ? (dstAddr - (int*)dstOrgAddr_char) : 0;

  /* get the descriptor of atom (remote coarray)
   */
  atomDesc = _XMPCO_get_descForMemoryChunk(cinfo);
  moldOrgAddr_char = _XMPCO_get_orgAddrOfMemoryChunk(cinfo);
  moldOffset = moldAddr - (int*)moldOrgAddr_char;

  int image = _XMPF_get_initial_image_withDescPtr(coindex, cinfo);

  /* action
   */
  _XMP_atomic_ref_1(atomDesc, moldOffset, image-1,
                    dstAddr, dstDesc, dstOffset, sizeof(int));
}

