#ifndef _XMPCO_PUTGET_H
#define _XMPCO_PUTGET_H

extern void _XMPCO_getVector_DMA(void *descPtr, char *baseAddr, int bytes, int coindex,
                                 void *descDMA, size_t offsetDMA, char *nameDMA);
extern void _XMPCO_getVector_buffer(void *descPtr, char *baseAddr, int bytesRU, int coindex,
                                    char *result, int bytes);



/*****************************************\
  TEMPORARY
\*****************************************/

// declared in ../../libxmp/include/xmp_func_decl.h
extern void _XMP_atomic_define_1(void *, size_t, int, int, void*, size_t, size_t);



#endif /*_XMPCO_PUTGET_H*/
