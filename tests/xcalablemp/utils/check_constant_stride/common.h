#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct _XMP_array_section{
  long long start;
  long long length;
  long long stride;
  long long elmts;
  long long distance;
} _XMP_array_section_t;


extern int _heavy_check_stride(_XMP_array_section_t*, int);
extern int _is_the_same_constant_stride(const _XMP_array_section_t *, const int);
extern unsigned int _XMP_get_dim_of_allelmts(const int, const _XMP_array_section_t*);
extern int _check_continuous(const _XMP_array_section_t *, const int, const int);

#define _XMP_N_INT_FALSE false
#define _XMP_N_INT_TRUE  true

#endif
