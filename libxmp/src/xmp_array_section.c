#include "xmp_internal.h"

static void _XCALABLEMP_normalize_array_section(int *lower, int *upper, int *stride);

static void _XCALABLEMP_normalize_array_section(int *lower, int *upper, int *stride) {
  // setup temporary variables
  int l, u;
  int s = *(stride);
  if (s > 0) {
    l = *lower;
    u = *upper;
  }
  else if (s < 0) {
    l = *upper;
    u = *lower;
  }
  else {
    _XCALABLEMP_fatal("the stride of <array-section> is 0");
  }

  // normalize values
  if (s > 0) {
    u = u - ((u - l) % s);
    *upper = u;
  }
  else {
    s = -s;
    l = l + ((u - l) % s);
    *lower = l;
    *upper = u;
    *stride = s;
  }
}

// ----- pack array
// --- dimension 1
#define _XCALABLEMP_SM_PACK_ARRAY_1(_type) \
(_type *buf_addr, const _type *const src_addr, \
 int src_lower, int src_upper, int src_stride) { \
  _XCALABLEMP_normalize_array_section(&src_lower, &src_upper, &src_stride); \
  for (int i = src_lower; i <= src_upper; i += src_stride) { \
    *buf_addr = src_addr[i]; \
    buf_addr++; \
  } \
}

void _XCALABLEMP_pack_array_1_BOOL			_XCALABLEMP_SM_PACK_ARRAY_1(_Bool)
void _XCALABLEMP_pack_array_1_CHAR			_XCALABLEMP_SM_PACK_ARRAY_1(char)
void _XCALABLEMP_pack_array_1_UNSIGNED_CHAR		_XCALABLEMP_SM_PACK_ARRAY_1(unsigned char)
void _XCALABLEMP_pack_array_1_SHORT			_XCALABLEMP_SM_PACK_ARRAY_1(short)
void _XCALABLEMP_pack_array_1_UNSIGNED_SHORT		_XCALABLEMP_SM_PACK_ARRAY_1(unsigned short)
void _XCALABLEMP_pack_array_1_INT			_XCALABLEMP_SM_PACK_ARRAY_1(int)
void _XCALABLEMP_pack_array_1_UNSIGNED_INT		_XCALABLEMP_SM_PACK_ARRAY_1(unsigned int)
void _XCALABLEMP_pack_array_1_LONG			_XCALABLEMP_SM_PACK_ARRAY_1(long)
void _XCALABLEMP_pack_array_1_UNSIGNED_LONG		_XCALABLEMP_SM_PACK_ARRAY_1(unsigned long)
void _XCALABLEMP_pack_array_1_LONG_LONG			_XCALABLEMP_SM_PACK_ARRAY_1(long long)
void _XCALABLEMP_pack_array_1_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_PACK_ARRAY_1(unsigned long long)
void _XCALABLEMP_pack_array_1_FLOAT			_XCALABLEMP_SM_PACK_ARRAY_1(float)
void _XCALABLEMP_pack_array_1_DOUBLE			_XCALABLEMP_SM_PACK_ARRAY_1(double)
void _XCALABLEMP_pack_array_1_LONG_DOUBLE		_XCALABLEMP_SM_PACK_ARRAY_1(long double)

// --- dimension 2
#define _XCALABLEMP_SM_PACK_ARRAY_2(_type) \
(_type *buf_addr, const _type *const src_addr, \
 int src_lower0, int src_upper0, int src_stride0, const int src_dim_acc0, \
 int src_lower1, int src_upper1, int src_stride1) { \
  _XCALABLEMP_normalize_array_section(&src_lower0, &src_upper0, &src_stride0); \
  _XCALABLEMP_normalize_array_section(&src_lower1, &src_upper1, &src_stride1); \
  for (int j = src_lower1; j <= src_upper1; j += src_stride1) { \
    const _type *const addr = src_addr + (j * src_dim_acc0); \
    for (int i = src_lower0; i <= src_upper0; i += src_stride0) { \
      *buf_addr = addr[i]; \
      buf_addr++; \
    } \
  } \
}

void _XCALABLEMP_pack_array_2_BOOL			_XCALABLEMP_SM_PACK_ARRAY_2(_Bool)
void _XCALABLEMP_pack_array_2_CHAR			_XCALABLEMP_SM_PACK_ARRAY_2(char)
void _XCALABLEMP_pack_array_2_UNSIGNED_CHAR		_XCALABLEMP_SM_PACK_ARRAY_2(unsigned char)
void _XCALABLEMP_pack_array_2_SHORT			_XCALABLEMP_SM_PACK_ARRAY_2(short)
void _XCALABLEMP_pack_array_2_UNSIGNED_SHORT		_XCALABLEMP_SM_PACK_ARRAY_2(unsigned short)
void _XCALABLEMP_pack_array_2_INT			_XCALABLEMP_SM_PACK_ARRAY_2(int)
void _XCALABLEMP_pack_array_2_UNSIGNED_INT		_XCALABLEMP_SM_PACK_ARRAY_2(unsigned int)
void _XCALABLEMP_pack_array_2_LONG			_XCALABLEMP_SM_PACK_ARRAY_2(long)
void _XCALABLEMP_pack_array_2_UNSIGNED_LONG		_XCALABLEMP_SM_PACK_ARRAY_2(unsigned long)
void _XCALABLEMP_pack_array_2_LONG_LONG			_XCALABLEMP_SM_PACK_ARRAY_2(long long)
void _XCALABLEMP_pack_array_2_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_PACK_ARRAY_2(unsigned long long)
void _XCALABLEMP_pack_array_2_FLOAT			_XCALABLEMP_SM_PACK_ARRAY_2(float)
void _XCALABLEMP_pack_array_2_DOUBLE			_XCALABLEMP_SM_PACK_ARRAY_2(double)
void _XCALABLEMP_pack_array_2_LONG_DOUBLE		_XCALABLEMP_SM_PACK_ARRAY_2(long double)

// --- dimension 3
#define _XCALABLEMP_SM_PACK_ARRAY_3(_type) \
(_type *buf_addr, const _type *const src_addr, \
 int src_lower0, int src_upper0, int src_stride0, const int src_dim_acc0, \
 int src_lower1, int src_upper1, int src_stride1, const int src_dim_acc1, \
 int src_lower2, int src_upper2, int src_stride2) { \
  _XCALABLEMP_normalize_array_section(&src_lower0, &src_upper0, &src_stride0); \
  _XCALABLEMP_normalize_array_section(&src_lower1, &src_upper1, &src_stride1); \
  _XCALABLEMP_normalize_array_section(&src_lower2, &src_upper2, &src_stride2); \
  for (int k = src_lower2; k <= src_upper2; k += src_stride2) { \
    for (int j = src_lower1; j <= src_upper1; j += src_stride1) { \
      const _type *const addr = src_addr + (j * src_dim_acc0) + (k * src_dim_acc1); \
      for (int i = src_lower0; i <= src_upper0; i += src_stride0) { \
        *buf_addr = addr[i]; \
        buf_addr++; \
      } \
    } \
  } \
}

void _XCALABLEMP_pack_array_3_BOOL			_XCALABLEMP_SM_PACK_ARRAY_3(_Bool)
void _XCALABLEMP_pack_array_3_CHAR			_XCALABLEMP_SM_PACK_ARRAY_3(char)
void _XCALABLEMP_pack_array_3_UNSIGNED_CHAR		_XCALABLEMP_SM_PACK_ARRAY_3(unsigned char)
void _XCALABLEMP_pack_array_3_SHORT			_XCALABLEMP_SM_PACK_ARRAY_3(short)
void _XCALABLEMP_pack_array_3_UNSIGNED_SHORT		_XCALABLEMP_SM_PACK_ARRAY_3(unsigned short)
void _XCALABLEMP_pack_array_3_INT			_XCALABLEMP_SM_PACK_ARRAY_3(int)
void _XCALABLEMP_pack_array_3_UNSIGNED_INT		_XCALABLEMP_SM_PACK_ARRAY_3(unsigned int)
void _XCALABLEMP_pack_array_3_LONG			_XCALABLEMP_SM_PACK_ARRAY_3(long)
void _XCALABLEMP_pack_array_3_UNSIGNED_LONG		_XCALABLEMP_SM_PACK_ARRAY_3(unsigned long)
void _XCALABLEMP_pack_array_3_LONG_LONG			_XCALABLEMP_SM_PACK_ARRAY_3(long long)
void _XCALABLEMP_pack_array_3_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_PACK_ARRAY_3(unsigned long long)
void _XCALABLEMP_pack_array_3_FLOAT			_XCALABLEMP_SM_PACK_ARRAY_3(float)
void _XCALABLEMP_pack_array_3_DOUBLE			_XCALABLEMP_SM_PACK_ARRAY_3(double)
void _XCALABLEMP_pack_array_3_LONG_DOUBLE		_XCALABLEMP_SM_PACK_ARRAY_3(long double)

// --- dimension 4
#define _XCALABLEMP_SM_PACK_ARRAY_4(_type) \
(_type *buf_addr, const _type *const src_addr, \
 int src_lower0, int src_upper0, int src_stride0, const int src_dim_acc0, \
 int src_lower1, int src_upper1, int src_stride1, const int src_dim_acc1, \
 int src_lower2, int src_upper2, int src_stride2, const int src_dim_acc2, \
 int src_lower3, int src_upper3, int src_stride3) { \
  _XCALABLEMP_normalize_array_section(&src_lower0, &src_upper0, &src_stride0); \
  _XCALABLEMP_normalize_array_section(&src_lower1, &src_upper1, &src_stride1); \
  _XCALABLEMP_normalize_array_section(&src_lower2, &src_upper2, &src_stride2); \
  _XCALABLEMP_normalize_array_section(&src_lower3, &src_upper3, &src_stride3); \
  for (int l = src_lower3; l <= src_upper3; l += src_stride3) { \
    for (int k = src_lower2; k <= src_upper2; k += src_stride2) { \
      for (int j = src_lower1; j <= src_upper1; j += src_stride1) { \
        const _type *const addr = src_addr + (j * src_dim_acc0) + (k * src_dim_acc1); + (l * src_dim_acc2); \
        for (int i = src_lower0; i <= src_upper0; i += src_stride0) { \
          *buf_addr = addr[i]; \
          buf_addr++; \
        } \
      } \
    } \
  } \
}

void _XCALABLEMP_pack_array_4_BOOL			_XCALABLEMP_SM_PACK_ARRAY_4(_Bool)
void _XCALABLEMP_pack_array_4_CHAR			_XCALABLEMP_SM_PACK_ARRAY_4(char)
void _XCALABLEMP_pack_array_4_UNSIGNED_CHAR		_XCALABLEMP_SM_PACK_ARRAY_4(unsigned char)
void _XCALABLEMP_pack_array_4_SHORT			_XCALABLEMP_SM_PACK_ARRAY_4(short)
void _XCALABLEMP_pack_array_4_UNSIGNED_SHORT		_XCALABLEMP_SM_PACK_ARRAY_4(unsigned short)
void _XCALABLEMP_pack_array_4_INT			_XCALABLEMP_SM_PACK_ARRAY_4(int)
void _XCALABLEMP_pack_array_4_UNSIGNED_INT		_XCALABLEMP_SM_PACK_ARRAY_4(unsigned int)
void _XCALABLEMP_pack_array_4_LONG			_XCALABLEMP_SM_PACK_ARRAY_4(long)
void _XCALABLEMP_pack_array_4_UNSIGNED_LONG		_XCALABLEMP_SM_PACK_ARRAY_4(unsigned long)
void _XCALABLEMP_pack_array_4_LONG_LONG			_XCALABLEMP_SM_PACK_ARRAY_4(long long)
void _XCALABLEMP_pack_array_4_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_PACK_ARRAY_4(unsigned long long)
void _XCALABLEMP_pack_array_4_FLOAT			_XCALABLEMP_SM_PACK_ARRAY_4(float)
void _XCALABLEMP_pack_array_4_DOUBLE			_XCALABLEMP_SM_PACK_ARRAY_4(double)
void _XCALABLEMP_pack_array_4_LONG_DOUBLE		_XCALABLEMP_SM_PACK_ARRAY_4(long double)

// --- dimension 5
#define _XCALABLEMP_SM_PACK_ARRAY_5(_type) \
(_type *buf_addr, const _type *const src_addr, \
 int src_lower0, int src_upper0, int src_stride0, const int src_dim_acc0, \
 int src_lower1, int src_upper1, int src_stride1, const int src_dim_acc1, \
 int src_lower2, int src_upper2, int src_stride2, const int src_dim_acc2, \
 int src_lower3, int src_upper3, int src_stride3, const int src_dim_acc3, \
 int src_lower4, int src_upper4, int src_stride4) { \
  _XCALABLEMP_normalize_array_section(&src_lower0, &src_upper0, &src_stride0); \
  _XCALABLEMP_normalize_array_section(&src_lower1, &src_upper1, &src_stride1); \
  _XCALABLEMP_normalize_array_section(&src_lower2, &src_upper2, &src_stride2); \
  _XCALABLEMP_normalize_array_section(&src_lower3, &src_upper3, &src_stride3); \
  _XCALABLEMP_normalize_array_section(&src_lower4, &src_upper4, &src_stride4); \
  for (int m = src_lower4; m <= src_upper4; m += src_stride4) { \
    for (int l = src_lower3; l <= src_upper3; l += src_stride3) { \
      for (int k = src_lower2; k <= src_upper2; k += src_stride2) { \
        for (int j = src_lower1; j <= src_upper1; j += src_stride1) { \
          const _type *const addr = src_addr + (j * src_dim_acc0) + (k * src_dim_acc1); + (l * src_dim_acc2) + (m * src_dim_acc3); \
          for (int i = src_lower0; i <= src_upper0; i += src_stride0) { \
            *buf_addr = addr[i]; \
            buf_addr++; \
          } \
        } \
      } \
    } \
  } \
}

void _XCALABLEMP_pack_array_5_BOOL			_XCALABLEMP_SM_PACK_ARRAY_5(_Bool)
void _XCALABLEMP_pack_array_5_CHAR			_XCALABLEMP_SM_PACK_ARRAY_5(char)
void _XCALABLEMP_pack_array_5_UNSIGNED_CHAR		_XCALABLEMP_SM_PACK_ARRAY_5(unsigned char)
void _XCALABLEMP_pack_array_5_SHORT			_XCALABLEMP_SM_PACK_ARRAY_5(short)
void _XCALABLEMP_pack_array_5_UNSIGNED_SHORT		_XCALABLEMP_SM_PACK_ARRAY_5(unsigned short)
void _XCALABLEMP_pack_array_5_INT			_XCALABLEMP_SM_PACK_ARRAY_5(int)
void _XCALABLEMP_pack_array_5_UNSIGNED_INT		_XCALABLEMP_SM_PACK_ARRAY_5(unsigned int)
void _XCALABLEMP_pack_array_5_LONG			_XCALABLEMP_SM_PACK_ARRAY_5(long)
void _XCALABLEMP_pack_array_5_UNSIGNED_LONG		_XCALABLEMP_SM_PACK_ARRAY_5(unsigned long)
void _XCALABLEMP_pack_array_5_LONG_LONG			_XCALABLEMP_SM_PACK_ARRAY_5(long long)
void _XCALABLEMP_pack_array_5_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_PACK_ARRAY_5(unsigned long long)
void _XCALABLEMP_pack_array_5_FLOAT			_XCALABLEMP_SM_PACK_ARRAY_5(float)
void _XCALABLEMP_pack_array_5_DOUBLE			_XCALABLEMP_SM_PACK_ARRAY_5(double)
void _XCALABLEMP_pack_array_5_LONG_DOUBLE		_XCALABLEMP_SM_PACK_ARRAY_5(long double)

// --- dimension 6
#define _XCALABLEMP_SM_PACK_ARRAY_6(_type) \
(_type *buf_addr, const _type *const src_addr, \
 int src_lower0, int src_upper0, int src_stride0, const int src_dim_acc0, \
 int src_lower1, int src_upper1, int src_stride1, const int src_dim_acc1, \
 int src_lower2, int src_upper2, int src_stride2, const int src_dim_acc2, \
 int src_lower3, int src_upper3, int src_stride3, const int src_dim_acc3, \
 int src_lower4, int src_upper4, int src_stride4, const int src_dim_acc4, \
 int src_lower5, int src_upper5, int src_stride5) { \
  _XCALABLEMP_normalize_array_section(&src_lower0, &src_upper0, &src_stride0); \
  _XCALABLEMP_normalize_array_section(&src_lower1, &src_upper1, &src_stride1); \
  _XCALABLEMP_normalize_array_section(&src_lower2, &src_upper2, &src_stride2); \
  _XCALABLEMP_normalize_array_section(&src_lower3, &src_upper3, &src_stride3); \
  _XCALABLEMP_normalize_array_section(&src_lower4, &src_upper4, &src_stride4); \
  _XCALABLEMP_normalize_array_section(&src_lower5, &src_upper5, &src_stride5); \
  for (int n = src_lower5; n <= src_upper5; n += src_stride5) { \
    for (int m = src_lower4; m <= src_upper4; m += src_stride4) { \
      for (int l = src_lower3; l <= src_upper3; l += src_stride3) { \
        for (int k = src_lower2; k <= src_upper2; k += src_stride2) { \
          for (int j = src_lower1; j <= src_upper1; j += src_stride1) { \
            const _type *const addr = src_addr + (j * src_dim_acc0) + (k * src_dim_acc1); + (l * src_dim_acc2) + \
                                                 (m * src_dim_acc3) + (n * src_dim_acc4); \
            for (int i = src_lower0; i <= src_upper0; i += src_stride0) { \
              *buf_addr = addr[i]; \
              buf_addr++; \
            } \
          } \
        } \
      } \
    } \
  } \
}

void _XCALABLEMP_pack_array_6_BOOL			_XCALABLEMP_SM_PACK_ARRAY_6(_Bool)
void _XCALABLEMP_pack_array_6_CHAR			_XCALABLEMP_SM_PACK_ARRAY_6(char)
void _XCALABLEMP_pack_array_6_UNSIGNED_CHAR		_XCALABLEMP_SM_PACK_ARRAY_6(unsigned char)
void _XCALABLEMP_pack_array_6_SHORT			_XCALABLEMP_SM_PACK_ARRAY_6(short)
void _XCALABLEMP_pack_array_6_UNSIGNED_SHORT		_XCALABLEMP_SM_PACK_ARRAY_6(unsigned short)
void _XCALABLEMP_pack_array_6_INT			_XCALABLEMP_SM_PACK_ARRAY_6(int)
void _XCALABLEMP_pack_array_6_UNSIGNED_INT		_XCALABLEMP_SM_PACK_ARRAY_6(unsigned int)
void _XCALABLEMP_pack_array_6_LONG			_XCALABLEMP_SM_PACK_ARRAY_6(long)
void _XCALABLEMP_pack_array_6_UNSIGNED_LONG		_XCALABLEMP_SM_PACK_ARRAY_6(unsigned long)
void _XCALABLEMP_pack_array_6_LONG_LONG			_XCALABLEMP_SM_PACK_ARRAY_6(long long)
void _XCALABLEMP_pack_array_6_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_PACK_ARRAY_6(unsigned long long)
void _XCALABLEMP_pack_array_6_FLOAT			_XCALABLEMP_SM_PACK_ARRAY_6(float)
void _XCALABLEMP_pack_array_6_DOUBLE			_XCALABLEMP_SM_PACK_ARRAY_6(double)
void _XCALABLEMP_pack_array_6_LONG_DOUBLE		_XCALABLEMP_SM_PACK_ARRAY_6(long double)

// --- dimension 7
#define _XCALABLEMP_SM_PACK_ARRAY_7(_type) \
(_type *buf_addr, const _type *const src_addr, \
 int src_lower0, int src_upper0, int src_stride0, const int src_dim_acc0, \
 int src_lower1, int src_upper1, int src_stride1, const int src_dim_acc1, \
 int src_lower2, int src_upper2, int src_stride2, const int src_dim_acc2, \
 int src_lower3, int src_upper3, int src_stride3, const int src_dim_acc3, \
 int src_lower4, int src_upper4, int src_stride4, const int src_dim_acc4, \
 int src_lower5, int src_upper5, int src_stride5, const int src_dim_acc5, \
 int src_lower6, int src_upper6, int src_stride6) { \
  _XCALABLEMP_normalize_array_section(&src_lower0, &src_upper0, &src_stride0); \
  _XCALABLEMP_normalize_array_section(&src_lower1, &src_upper1, &src_stride1); \
  _XCALABLEMP_normalize_array_section(&src_lower2, &src_upper2, &src_stride2); \
  _XCALABLEMP_normalize_array_section(&src_lower3, &src_upper3, &src_stride3); \
  _XCALABLEMP_normalize_array_section(&src_lower4, &src_upper4, &src_stride4); \
  _XCALABLEMP_normalize_array_section(&src_lower5, &src_upper5, &src_stride5); \
  _XCALABLEMP_normalize_array_section(&src_lower6, &src_upper6, &src_stride6); \
  for (int o = src_lower6; o <= src_upper6; o += src_stride6) { \
    for (int n = src_lower5; n <= src_upper5; n += src_stride5) { \
      for (int m = src_lower4; m <= src_upper4; m += src_stride4) { \
        for (int l = src_lower3; l <= src_upper3; l += src_stride3) { \
          for (int k = src_lower2; k <= src_upper2; k += src_stride2) { \
            for (int j = src_lower1; j <= src_upper1; j += src_stride1) { \
              const _type *const addr = src_addr + (j * src_dim_acc0) + (k * src_dim_acc1); + (l * src_dim_acc2) + \
                                                   (m * src_dim_acc3) + (n * src_dim_acc4); + (o * src_dim_acc5); \
              for (int i = src_lower0; i <= src_upper0; i += src_stride0) { \
                *buf_addr = addr[i]; \
                buf_addr++; \
              } \
            } \
          } \
        } \
      } \
    } \
  } \
}

void _XCALABLEMP_pack_array_7_BOOL			_XCALABLEMP_SM_PACK_ARRAY_7(_Bool)
void _XCALABLEMP_pack_array_7_CHAR			_XCALABLEMP_SM_PACK_ARRAY_7(char)
void _XCALABLEMP_pack_array_7_UNSIGNED_CHAR		_XCALABLEMP_SM_PACK_ARRAY_7(unsigned char)
void _XCALABLEMP_pack_array_7_SHORT			_XCALABLEMP_SM_PACK_ARRAY_7(short)
void _XCALABLEMP_pack_array_7_UNSIGNED_SHORT		_XCALABLEMP_SM_PACK_ARRAY_7(unsigned short)
void _XCALABLEMP_pack_array_7_INT			_XCALABLEMP_SM_PACK_ARRAY_7(int)
void _XCALABLEMP_pack_array_7_UNSIGNED_INT		_XCALABLEMP_SM_PACK_ARRAY_7(unsigned int)
void _XCALABLEMP_pack_array_7_LONG			_XCALABLEMP_SM_PACK_ARRAY_7(long)
void _XCALABLEMP_pack_array_7_UNSIGNED_LONG		_XCALABLEMP_SM_PACK_ARRAY_7(unsigned long)
void _XCALABLEMP_pack_array_7_LONG_LONG			_XCALABLEMP_SM_PACK_ARRAY_7(long long)
void _XCALABLEMP_pack_array_7_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_PACK_ARRAY_7(unsigned long long)
void _XCALABLEMP_pack_array_7_FLOAT			_XCALABLEMP_SM_PACK_ARRAY_7(float)
void _XCALABLEMP_pack_array_7_DOUBLE			_XCALABLEMP_SM_PACK_ARRAY_7(double)
void _XCALABLEMP_pack_array_7_LONG_DOUBLE		_XCALABLEMP_SM_PACK_ARRAY_7(long double)


// ----- unpack array
// --- dimension 1
#define _XCALABLEMP_SM_UNPACK_ARRAY_1(_type) \
(_type *const dst_addr, _type *buf_addr, \
 int dst_lower, int dst_upper, int dst_stride) { \
  _XCALABLEMP_normalize_array_section(&dst_lower, &dst_upper, &dst_stride); \
  for (int i = dst_lower; i <= dst_upper; i += dst_stride) { \
    dst_addr[i] = *buf_addr; \
    buf_addr++; \
  } \
}

void _XCALABLEMP_unpack_array_1_BOOL			_XCALABLEMP_SM_UNPACK_ARRAY_1(_Bool)
void _XCALABLEMP_unpack_array_1_CHAR			_XCALABLEMP_SM_UNPACK_ARRAY_1(char)
void _XCALABLEMP_unpack_array_1_UNSIGNED_CHAR		_XCALABLEMP_SM_UNPACK_ARRAY_1(unsigned char)
void _XCALABLEMP_unpack_array_1_SHORT			_XCALABLEMP_SM_UNPACK_ARRAY_1(short)
void _XCALABLEMP_unpack_array_1_UNSIGNED_SHORT		_XCALABLEMP_SM_UNPACK_ARRAY_1(unsigned short)
void _XCALABLEMP_unpack_array_1_INT			_XCALABLEMP_SM_UNPACK_ARRAY_1(int)
void _XCALABLEMP_unpack_array_1_UNSIGNED_INT		_XCALABLEMP_SM_UNPACK_ARRAY_1(unsigned int)
void _XCALABLEMP_unpack_array_1_LONG			_XCALABLEMP_SM_UNPACK_ARRAY_1(long)
void _XCALABLEMP_unpack_array_1_UNSIGNED_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_1(unsigned long)
void _XCALABLEMP_unpack_array_1_LONG_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_1(long long)
void _XCALABLEMP_unpack_array_1_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_UNPACK_ARRAY_1(unsigned long long)
void _XCALABLEMP_unpack_array_1_FLOAT			_XCALABLEMP_SM_UNPACK_ARRAY_1(float)
void _XCALABLEMP_unpack_array_1_DOUBLE			_XCALABLEMP_SM_UNPACK_ARRAY_1(double)
void _XCALABLEMP_unpack_array_1_LONG_DOUBLE		_XCALABLEMP_SM_UNPACK_ARRAY_1(long double)

// --- dimension 2
#define _XCALABLEMP_SM_UNPACK_ARRAY_2(_type) \
(_type *const dst_addr, _type *buf_addr, \
 int dst_lower0, int dst_upper0, int dst_stride0, const int dst_dim_acc0, \
 int dst_lower1, int dst_upper1, int dst_stride1) { \
  _XCALABLEMP_normalize_array_section(&dst_lower0, &dst_upper0, &dst_stride0); \
  _XCALABLEMP_normalize_array_section(&dst_lower1, &dst_upper1, &dst_stride1); \
  for (int j = dst_lower1; j <= dst_upper1; j += dst_stride1) { \
    _type *const addr = dst_addr + (j * dst_dim_acc0); \
    for (int i = dst_lower0; i <= dst_upper0; i += dst_stride0) { \
      addr[i] = *buf_addr; \
      buf_addr++; \
    } \
  } \
}

void _XCALABLEMP_unpack_array_2_BOOL			_XCALABLEMP_SM_UNPACK_ARRAY_2(_Bool)
void _XCALABLEMP_unpack_array_2_CHAR			_XCALABLEMP_SM_UNPACK_ARRAY_2(char)
void _XCALABLEMP_unpack_array_2_UNSIGNED_CHAR		_XCALABLEMP_SM_UNPACK_ARRAY_2(unsigned char)
void _XCALABLEMP_unpack_array_2_SHORT			_XCALABLEMP_SM_UNPACK_ARRAY_2(short)
void _XCALABLEMP_unpack_array_2_UNSIGNED_SHORT		_XCALABLEMP_SM_UNPACK_ARRAY_2(unsigned short)
void _XCALABLEMP_unpack_array_2_INT			_XCALABLEMP_SM_UNPACK_ARRAY_2(int)
void _XCALABLEMP_unpack_array_2_UNSIGNED_INT		_XCALABLEMP_SM_UNPACK_ARRAY_2(unsigned int)
void _XCALABLEMP_unpack_array_2_LONG			_XCALABLEMP_SM_UNPACK_ARRAY_2(long)
void _XCALABLEMP_unpack_array_2_UNSIGNED_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_2(unsigned long)
void _XCALABLEMP_unpack_array_2_LONG_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_2(long long)
void _XCALABLEMP_unpack_array_2_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_UNPACK_ARRAY_2(unsigned long long)
void _XCALABLEMP_unpack_array_2_FLOAT			_XCALABLEMP_SM_UNPACK_ARRAY_2(float)
void _XCALABLEMP_unpack_array_2_DOUBLE			_XCALABLEMP_SM_UNPACK_ARRAY_2(double)
void _XCALABLEMP_unpack_array_2_LONG_DOUBLE		_XCALABLEMP_SM_UNPACK_ARRAY_2(long double)

// --- dimension 3
#define _XCALABLEMP_SM_UNPACK_ARRAY_3(_type) \
(_type *const dst_addr, _type *buf_addr, \
 int dst_lower0, int dst_upper0, int dst_stride0, const int dst_dim_acc0, \
 int dst_lower1, int dst_upper1, int dst_stride1, const int dst_dim_acc1, \
 int dst_lower2, int dst_upper2, int dst_stride2) { \
  _XCALABLEMP_normalize_array_section(&dst_lower0, &dst_upper0, &dst_stride0); \
  _XCALABLEMP_normalize_array_section(&dst_lower1, &dst_upper1, &dst_stride1); \
  _XCALABLEMP_normalize_array_section(&dst_lower2, &dst_upper2, &dst_stride2); \
  for (int k = dst_lower2; k <= dst_upper2; k += dst_stride2) { \
    for (int j = dst_lower1; j <= dst_upper1; j += dst_stride1) { \
      _type *const addr = dst_addr + (j * dst_dim_acc0) + (k * dst_dim_acc1); \
      for (int i = dst_lower0; i <= dst_upper0; i += dst_stride0) { \
        addr[i] = *buf_addr; \
        buf_addr++; \
      } \
    } \
  } \
}

void _XCALABLEMP_unpack_array_3_BOOL			_XCALABLEMP_SM_UNPACK_ARRAY_3(_Bool)
void _XCALABLEMP_unpack_array_3_CHAR			_XCALABLEMP_SM_UNPACK_ARRAY_3(char)
void _XCALABLEMP_unpack_array_3_UNSIGNED_CHAR		_XCALABLEMP_SM_UNPACK_ARRAY_3(unsigned char)
void _XCALABLEMP_unpack_array_3_SHORT			_XCALABLEMP_SM_UNPACK_ARRAY_3(short)
void _XCALABLEMP_unpack_array_3_UNSIGNED_SHORT		_XCALABLEMP_SM_UNPACK_ARRAY_3(unsigned short)
void _XCALABLEMP_unpack_array_3_INT			_XCALABLEMP_SM_UNPACK_ARRAY_3(int)
void _XCALABLEMP_unpack_array_3_UNSIGNED_INT		_XCALABLEMP_SM_UNPACK_ARRAY_3(unsigned int)
void _XCALABLEMP_unpack_array_3_LONG			_XCALABLEMP_SM_UNPACK_ARRAY_3(long)
void _XCALABLEMP_unpack_array_3_UNSIGNED_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_3(unsigned long)
void _XCALABLEMP_unpack_array_3_LONG_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_3(long long)
void _XCALABLEMP_unpack_array_3_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_UNPACK_ARRAY_3(unsigned long long)
void _XCALABLEMP_unpack_array_3_FLOAT			_XCALABLEMP_SM_UNPACK_ARRAY_3(float)
void _XCALABLEMP_unpack_array_3_DOUBLE			_XCALABLEMP_SM_UNPACK_ARRAY_3(double)
void _XCALABLEMP_unpack_array_3_LONG_DOUBLE		_XCALABLEMP_SM_UNPACK_ARRAY_3(long double)

// --- dimension 4
#define _XCALABLEMP_SM_UNPACK_ARRAY_4(_type) \
(_type *const dst_addr, _type *buf_addr, \
 int dst_lower0, int dst_upper0, int dst_stride0, const int dst_dim_acc0, \
 int dst_lower1, int dst_upper1, int dst_stride1, const int dst_dim_acc1, \
 int dst_lower2, int dst_upper2, int dst_stride2, const int dst_dim_acc2, \
 int dst_lower3, int dst_upper3, int dst_stride3) { \
  _XCALABLEMP_normalize_array_section(&dst_lower0, &dst_upper0, &dst_stride0); \
  _XCALABLEMP_normalize_array_section(&dst_lower1, &dst_upper1, &dst_stride1); \
  _XCALABLEMP_normalize_array_section(&dst_lower2, &dst_upper2, &dst_stride2); \
  _XCALABLEMP_normalize_array_section(&dst_lower3, &dst_upper3, &dst_stride3); \
  for (int l = dst_lower3; l <= dst_upper3; l += dst_stride3) { \
    for (int k = dst_lower2; k <= dst_upper2; k += dst_stride2) { \
      for (int j = dst_lower1; j <= dst_upper1; j += dst_stride1) { \
        _type *const addr = dst_addr + (j * dst_dim_acc0) + (k * dst_dim_acc1) + (l * dst_dim_acc2); \
        for (int i = dst_lower0; i <= dst_upper0; i += dst_stride0) { \
          addr[i] = *buf_addr; \
          buf_addr++; \
        } \
      } \
    } \
  } \
}

void _XCALABLEMP_unpack_array_4_BOOL			_XCALABLEMP_SM_UNPACK_ARRAY_4(_Bool)
void _XCALABLEMP_unpack_array_4_CHAR			_XCALABLEMP_SM_UNPACK_ARRAY_4(char)
void _XCALABLEMP_unpack_array_4_UNSIGNED_CHAR		_XCALABLEMP_SM_UNPACK_ARRAY_4(unsigned char)
void _XCALABLEMP_unpack_array_4_SHORT			_XCALABLEMP_SM_UNPACK_ARRAY_4(short)
void _XCALABLEMP_unpack_array_4_UNSIGNED_SHORT		_XCALABLEMP_SM_UNPACK_ARRAY_4(unsigned short)
void _XCALABLEMP_unpack_array_4_INT			_XCALABLEMP_SM_UNPACK_ARRAY_4(int)
void _XCALABLEMP_unpack_array_4_UNSIGNED_INT		_XCALABLEMP_SM_UNPACK_ARRAY_4(unsigned int)
void _XCALABLEMP_unpack_array_4_LONG			_XCALABLEMP_SM_UNPACK_ARRAY_4(long)
void _XCALABLEMP_unpack_array_4_UNSIGNED_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_4(unsigned long)
void _XCALABLEMP_unpack_array_4_LONG_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_4(long long)
void _XCALABLEMP_unpack_array_4_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_UNPACK_ARRAY_4(unsigned long long)
void _XCALABLEMP_unpack_array_4_FLOAT			_XCALABLEMP_SM_UNPACK_ARRAY_4(float)
void _XCALABLEMP_unpack_array_4_DOUBLE			_XCALABLEMP_SM_UNPACK_ARRAY_4(double)
void _XCALABLEMP_unpack_array_4_LONG_DOUBLE		_XCALABLEMP_SM_UNPACK_ARRAY_4(long double)

// --- dimension 5
#define _XCALABLEMP_SM_UNPACK_ARRAY_5(_type) \
(_type *const dst_addr, _type *buf_addr, \
 int dst_lower0, int dst_upper0, int dst_stride0, const int dst_dim_acc0, \
 int dst_lower1, int dst_upper1, int dst_stride1, const int dst_dim_acc1, \
 int dst_lower2, int dst_upper2, int dst_stride2, const int dst_dim_acc2, \
 int dst_lower3, int dst_upper3, int dst_stride3, const int dst_dim_acc3, \
 int dst_lower4, int dst_upper4, int dst_stride4) { \
  _XCALABLEMP_normalize_array_section(&dst_lower0, &dst_upper0, &dst_stride0); \
  _XCALABLEMP_normalize_array_section(&dst_lower1, &dst_upper1, &dst_stride1); \
  _XCALABLEMP_normalize_array_section(&dst_lower2, &dst_upper2, &dst_stride2); \
  _XCALABLEMP_normalize_array_section(&dst_lower3, &dst_upper3, &dst_stride3); \
  _XCALABLEMP_normalize_array_section(&dst_lower4, &dst_upper4, &dst_stride4); \
  for (int m = dst_lower4; m <= dst_upper4; m += dst_stride4) { \
    for (int l = dst_lower3; l <= dst_upper3; l += dst_stride3) { \
      for (int k = dst_lower2; k <= dst_upper2; k += dst_stride2) { \
        for (int j = dst_lower1; j <= dst_upper1; j += dst_stride1) { \
          _type *const addr = dst_addr + (j * dst_dim_acc0) + (k * dst_dim_acc1) + (l * dst_dim_acc2) + (m * dst_dim_acc3); \
          for (int i = dst_lower0; i <= dst_upper0; i += dst_stride0) { \
            addr[i] = *buf_addr; \
            buf_addr++; \
          } \
        } \
      } \
    } \
  } \
}

void _XCALABLEMP_unpack_array_5_BOOL			_XCALABLEMP_SM_UNPACK_ARRAY_5(_Bool)
void _XCALABLEMP_unpack_array_5_CHAR			_XCALABLEMP_SM_UNPACK_ARRAY_5(char)
void _XCALABLEMP_unpack_array_5_UNSIGNED_CHAR		_XCALABLEMP_SM_UNPACK_ARRAY_5(unsigned char)
void _XCALABLEMP_unpack_array_5_SHORT			_XCALABLEMP_SM_UNPACK_ARRAY_5(short)
void _XCALABLEMP_unpack_array_5_UNSIGNED_SHORT		_XCALABLEMP_SM_UNPACK_ARRAY_5(unsigned short)
void _XCALABLEMP_unpack_array_5_INT			_XCALABLEMP_SM_UNPACK_ARRAY_5(int)
void _XCALABLEMP_unpack_array_5_UNSIGNED_INT		_XCALABLEMP_SM_UNPACK_ARRAY_5(unsigned int)
void _XCALABLEMP_unpack_array_5_LONG			_XCALABLEMP_SM_UNPACK_ARRAY_5(long)
void _XCALABLEMP_unpack_array_5_UNSIGNED_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_5(unsigned long)
void _XCALABLEMP_unpack_array_5_LONG_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_5(long long)
void _XCALABLEMP_unpack_array_5_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_UNPACK_ARRAY_5(unsigned long long)
void _XCALABLEMP_unpack_array_5_FLOAT			_XCALABLEMP_SM_UNPACK_ARRAY_5(float)
void _XCALABLEMP_unpack_array_5_DOUBLE			_XCALABLEMP_SM_UNPACK_ARRAY_5(double)
void _XCALABLEMP_unpack_array_5_LONG_DOUBLE		_XCALABLEMP_SM_UNPACK_ARRAY_5(long double)

// --- dimension 6
#define _XCALABLEMP_SM_UNPACK_ARRAY_6(_type) \
(_type *const dst_addr, _type *buf_addr, \
 int dst_lower0, int dst_upper0, int dst_stride0, const int dst_dim_acc0, \
 int dst_lower1, int dst_upper1, int dst_stride1, const int dst_dim_acc1, \
 int dst_lower2, int dst_upper2, int dst_stride2, const int dst_dim_acc2, \
 int dst_lower3, int dst_upper3, int dst_stride3, const int dst_dim_acc3, \
 int dst_lower4, int dst_upper4, int dst_stride4, const int dst_dim_acc4, \
 int dst_lower5, int dst_upper5, int dst_stride5) { \
  _XCALABLEMP_normalize_array_section(&dst_lower0, &dst_upper0, &dst_stride0); \
  _XCALABLEMP_normalize_array_section(&dst_lower1, &dst_upper1, &dst_stride1); \
  _XCALABLEMP_normalize_array_section(&dst_lower2, &dst_upper2, &dst_stride2); \
  _XCALABLEMP_normalize_array_section(&dst_lower3, &dst_upper3, &dst_stride3); \
  _XCALABLEMP_normalize_array_section(&dst_lower4, &dst_upper4, &dst_stride4); \
  _XCALABLEMP_normalize_array_section(&dst_lower5, &dst_upper5, &dst_stride5); \
  for (int n = dst_lower5; n <= dst_upper5; n += dst_stride5) { \
    for (int m = dst_lower4; m <= dst_upper4; m += dst_stride4) { \
      for (int l = dst_lower3; l <= dst_upper3; l += dst_stride3) { \
        for (int k = dst_lower2; k <= dst_upper2; k += dst_stride2) { \
          for (int j = dst_lower1; j <= dst_upper1; j += dst_stride1) { \
            _type *const addr = dst_addr + (j * dst_dim_acc0) + (k * dst_dim_acc1) + (l * dst_dim_acc2) + \
                                           (m * dst_dim_acc3) + (n * dst_dim_acc4); \
            for (int i = dst_lower0; i <= dst_upper0; i += dst_stride0) { \
              addr[i] = *buf_addr; \
              buf_addr++; \
            } \
          } \
        } \
      } \
    } \
  } \
}

void _XCALABLEMP_unpack_array_6_BOOL			_XCALABLEMP_SM_UNPACK_ARRAY_6(_Bool)
void _XCALABLEMP_unpack_array_6_CHAR			_XCALABLEMP_SM_UNPACK_ARRAY_6(char)
void _XCALABLEMP_unpack_array_6_UNSIGNED_CHAR		_XCALABLEMP_SM_UNPACK_ARRAY_6(unsigned char)
void _XCALABLEMP_unpack_array_6_SHORT			_XCALABLEMP_SM_UNPACK_ARRAY_6(short)
void _XCALABLEMP_unpack_array_6_UNSIGNED_SHORT		_XCALABLEMP_SM_UNPACK_ARRAY_6(unsigned short)
void _XCALABLEMP_unpack_array_6_INT			_XCALABLEMP_SM_UNPACK_ARRAY_6(int)
void _XCALABLEMP_unpack_array_6_UNSIGNED_INT		_XCALABLEMP_SM_UNPACK_ARRAY_6(unsigned int)
void _XCALABLEMP_unpack_array_6_LONG			_XCALABLEMP_SM_UNPACK_ARRAY_6(long)
void _XCALABLEMP_unpack_array_6_UNSIGNED_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_6(unsigned long)
void _XCALABLEMP_unpack_array_6_LONG_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_6(long long)
void _XCALABLEMP_unpack_array_6_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_UNPACK_ARRAY_6(unsigned long long)
void _XCALABLEMP_unpack_array_6_FLOAT			_XCALABLEMP_SM_UNPACK_ARRAY_6(float)
void _XCALABLEMP_unpack_array_6_DOUBLE			_XCALABLEMP_SM_UNPACK_ARRAY_6(double)
void _XCALABLEMP_unpack_array_6_LONG_DOUBLE		_XCALABLEMP_SM_UNPACK_ARRAY_6(long double)

// --- dimension 7
#define _XCALABLEMP_SM_UNPACK_ARRAY_7(_type) \
(_type *const dst_addr, _type *buf_addr, \
 int dst_lower0, int dst_upper0, int dst_stride0, const int dst_dim_acc0, \
 int dst_lower1, int dst_upper1, int dst_stride1, const int dst_dim_acc1, \
 int dst_lower2, int dst_upper2, int dst_stride2, const int dst_dim_acc2, \
 int dst_lower3, int dst_upper3, int dst_stride3, const int dst_dim_acc3, \
 int dst_lower4, int dst_upper4, int dst_stride4, const int dst_dim_acc4, \
 int dst_lower5, int dst_upper5, int dst_stride5, const int dst_dim_acc5, \
 int dst_lower6, int dst_upper6, int dst_stride6) { \
  _XCALABLEMP_normalize_array_section(&dst_lower0, &dst_upper0, &dst_stride0); \
  _XCALABLEMP_normalize_array_section(&dst_lower1, &dst_upper1, &dst_stride1); \
  _XCALABLEMP_normalize_array_section(&dst_lower2, &dst_upper2, &dst_stride2); \
  _XCALABLEMP_normalize_array_section(&dst_lower3, &dst_upper3, &dst_stride3); \
  _XCALABLEMP_normalize_array_section(&dst_lower4, &dst_upper4, &dst_stride4); \
  _XCALABLEMP_normalize_array_section(&dst_lower5, &dst_upper5, &dst_stride5); \
  _XCALABLEMP_normalize_array_section(&dst_lower6, &dst_upper6, &dst_stride6); \
  for (int o = dst_lower6; o <= dst_upper6; o += dst_stride6) { \
    for (int n = dst_lower5; n <= dst_upper5; n += dst_stride5) { \
      for (int m = dst_lower4; m <= dst_upper4; m += dst_stride4) { \
        for (int l = dst_lower3; l <= dst_upper3; l += dst_stride3) { \
          for (int k = dst_lower2; k <= dst_upper2; k += dst_stride2) { \
            for (int j = dst_lower1; j <= dst_upper1; j += dst_stride1) { \
              _type *const addr = dst_addr + (j * dst_dim_acc0) + (k * dst_dim_acc1) + (l * dst_dim_acc2) + \
                                             (m * dst_dim_acc3) + (n * dst_dim_acc4) + (o * dst_dim_acc5); \
              for (int i = dst_lower0; i <= dst_upper0; i += dst_stride0) { \
                addr[i] = *buf_addr; \
                buf_addr++; \
              } \
            } \
          } \
        } \
      } \
    } \
  } \
}

void _XCALABLEMP_unpack_array_7_BOOL			_XCALABLEMP_SM_UNPACK_ARRAY_7(_Bool)
void _XCALABLEMP_unpack_array_7_CHAR			_XCALABLEMP_SM_UNPACK_ARRAY_7(char)
void _XCALABLEMP_unpack_array_7_UNSIGNED_CHAR		_XCALABLEMP_SM_UNPACK_ARRAY_7(unsigned char)
void _XCALABLEMP_unpack_array_7_SHORT			_XCALABLEMP_SM_UNPACK_ARRAY_7(short)
void _XCALABLEMP_unpack_array_7_UNSIGNED_SHORT		_XCALABLEMP_SM_UNPACK_ARRAY_7(unsigned short)
void _XCALABLEMP_unpack_array_7_INT			_XCALABLEMP_SM_UNPACK_ARRAY_7(int)
void _XCALABLEMP_unpack_array_7_UNSIGNED_INT		_XCALABLEMP_SM_UNPACK_ARRAY_7(unsigned int)
void _XCALABLEMP_unpack_array_7_LONG			_XCALABLEMP_SM_UNPACK_ARRAY_7(long)
void _XCALABLEMP_unpack_array_7_UNSIGNED_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_7(unsigned long)
void _XCALABLEMP_unpack_array_7_LONG_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_7(long long)
void _XCALABLEMP_unpack_array_7_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_UNPACK_ARRAY_7(unsigned long long)
void _XCALABLEMP_unpack_array_7_FLOAT			_XCALABLEMP_SM_UNPACK_ARRAY_7(float)
void _XCALABLEMP_unpack_array_7_DOUBLE			_XCALABLEMP_SM_UNPACK_ARRAY_7(double)
void _XCALABLEMP_unpack_array_7_LONG_DOUBLE		_XCALABLEMP_SM_UNPACK_ARRAY_7(long double)
