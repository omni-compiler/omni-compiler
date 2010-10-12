#ifndef _XCALABLEMP_ARRAY_SECTION
#define _XCALABLEMP_ARRAY_SECTION

// pack/unpack shadow
extern void _XCALABLEMP_pack_shadow_buffer(void *buffer, void *src,
                                           int array_type, int array_dim, int *l, int *u, int *s, unsigned long long *d);
extern void _XCALABLEMP_unpack_shadow_buffer(void *dst, void *buffer,
                                             int array_type, int array_dim, int *l, int *u, int *s, unsigned long long *d);

#endif // _XCALABLEMP_ARRAY_SECTION
