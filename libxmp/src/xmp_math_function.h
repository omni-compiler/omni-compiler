/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#ifndef _XCALABLEMP_MATH_FUNCTION
#define _XCALABLEMP_MATH_FUNCTION

// --- integer functions
// calculate ceil(a/b)
#define _XCALABLEMP_M_CEILi(a_, b_) (((a_) % (b_)) == 0 ? ((a_) / (b_)) : ((a_) / (b_)) + 1)
// calculate floor(a/b)
#define _XCALABLEMP_M_FLOORi(a_, b_) ((a_) / (b_))
#define _XCALABLEMP_M_COUNTi(a_, b_) ((b_) - (a_) + 1)
#define _XCALABLEMP_M_COUNT_TRIPLETi(l_, u_, s_) (_XCALABLEMP_M_FLOORi(((u_) - (l_)), s_) + 1)

// --- generic functions
#define _XCALABLEMP_M_MAX(a_, b_) ((a_) > (b_) ? (a_) : (b_))
#define _XCALABLEMP_M_MIN(a_, b_) ((a_) > (b_) ? (b_) : (a_))

// defined in xmp_math_function.c
extern int _XCALABLEMP_modi_ll_i(long long value, int cycle);
extern int _XCALABLEMP_modi_i_i(int value, int cycle);

#endif //_XCALABLEMP_MATH_FUNCTION
