#ifndef _XCALABLEMP_MATH_MACRO
#define _XCALABLEMP_MATH_MACRO

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

#endif //_XCALABLEMP_MATH_MACRO
