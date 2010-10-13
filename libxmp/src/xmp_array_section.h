#ifndef _XCALABLEMP_ARRAY_SECTION
#define _XCALABLEMP_ARRAY_SECTION

extern void _XCALABLEMP_normalize_array_section(int *lower, int *upper, int *stride);
extern void _XCALABLEMP_pack_array(void *buffer, void *src,
                                   int array_type, int array_dim, int *l, int *u, int *s, unsigned long long *d);
extern void _XCALABLEMP_unpack_array(void *dst, void *buffer,
                                     int array_type, int array_dim, int *l, int *u, int *s, unsigned long long *d);

#endif // _XCALABLEMP_ARRAY_SECTION
