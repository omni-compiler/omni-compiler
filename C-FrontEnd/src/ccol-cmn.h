/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file ccol-cmn.h
 */

#ifndef _CCOL_CMN_H_
#define _CCOL_CMN_H_

typedef void* CCOL_Data;

#ifdef CCOL_DEBUG_MEM
#   define ccol_Free(x) { printf("@ccol-free:0x%08x@\n", (unsigned int)(x)); fflush(stdout); free(x); } 
#else
#   define ccol_Free(x) free(x)
#endif

#ifdef MTRACE
#define ccol_MallocNoInit(sz) malloc((sz))
#define ccol_Malloc(sz) memset(malloc((sz)), 0, (sz))
#else
extern void* ccol_MallocNoInit(unsigned int sz);
extern void* ccol_Malloc(unsigned int sz);
#endif
extern void ccol_outOfMemoryHandler(void (*handler)(void));
extern char* ccol_strdup(const char *s, unsigned int maxlen);
extern int ccol_strstarts(const char *s, const char *token);

#endif /* _CCOL_CMN_H_ */
