#include "xmp_constant.h"
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
(_type *buf_addr, _type *src_addr, \
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
void _XCALABLEMP_pack_array_1_LONGLONG			_XCALABLEMP_SM_PACK_ARRAY_1(long long)
void _XCALABLEMP_pack_array_1_UNSIGNED_LONGLONG		_XCALABLEMP_SM_PACK_ARRAY_1(unsigned long long)
void _XCALABLEMP_pack_array_1_FLOAT			_XCALABLEMP_SM_PACK_ARRAY_1(float)
void _XCALABLEMP_pack_array_1_DOUBLE			_XCALABLEMP_SM_PACK_ARRAY_1(double)
void _XCALABLEMP_pack_array_1_LONG_DOUBLE		_XCALABLEMP_SM_PACK_ARRAY_1(long double)

// --- dimension 2
#define _XCALABLEMP_SM_PACK_ARRAY_2(_type) \
(_type *buf_addr, _type *src_addr, \
 int src_lower0, int src_upper0, int src_stride0, unsigned long long src_dim_acc0, \
 int src_lower1, int src_upper1, int src_stride1) { \
  _XCALABLEMP_normalize_array_section(&src_lower0, &src_upper0, &src_stride0); \
  _XCALABLEMP_normalize_array_section(&src_lower1, &src_upper1, &src_stride1); \
  for (int j = src_lower0; j <= src_upper0; j += src_stride0) { \
    _type *addr = src_addr + (j * src_dim_acc0); \
    for (int i = src_lower1; i <= src_upper1; i += src_stride1) { \
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
void _XCALABLEMP_pack_array_2_LONGLONG			_XCALABLEMP_SM_PACK_ARRAY_2(long long)
void _XCALABLEMP_pack_array_2_UNSIGNED_LONGLONG		_XCALABLEMP_SM_PACK_ARRAY_2(unsigned long long)
void _XCALABLEMP_pack_array_2_FLOAT			_XCALABLEMP_SM_PACK_ARRAY_2(float)
void _XCALABLEMP_pack_array_2_DOUBLE			_XCALABLEMP_SM_PACK_ARRAY_2(double)
void _XCALABLEMP_pack_array_2_LONG_DOUBLE		_XCALABLEMP_SM_PACK_ARRAY_2(long double)

// --- dimension 3
#define _XCALABLEMP_SM_PACK_ARRAY_3(_type) \
(_type *buf_addr, _type *src_addr, \
 int src_lower0, int src_upper0, int src_stride0, unsigned long long src_dim_acc0, \
 int src_lower1, int src_upper1, int src_stride1, unsigned long long src_dim_acc1, \
 int src_lower2, int src_upper2, int src_stride2) { \
  _XCALABLEMP_normalize_array_section(&src_lower0, &src_upper0, &src_stride0); \
  _XCALABLEMP_normalize_array_section(&src_lower1, &src_upper1, &src_stride1); \
  _XCALABLEMP_normalize_array_section(&src_lower2, &src_upper2, &src_stride2); \
  for (int k = src_lower0; k <= src_upper0; k += src_stride0) { \
    for (int j = src_lower1; j <= src_upper1; j += src_stride1) { \
      _type *addr = src_addr + (k * src_dim_acc0) + (j * src_dim_acc1); \
      for (int i = src_lower2; i <= src_upper2; i += src_stride2) { \
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
void _XCALABLEMP_pack_array_3_LONGLONG			_XCALABLEMP_SM_PACK_ARRAY_3(long long)
void _XCALABLEMP_pack_array_3_UNSIGNED_LONGLONG		_XCALABLEMP_SM_PACK_ARRAY_3(unsigned long long)
void _XCALABLEMP_pack_array_3_FLOAT			_XCALABLEMP_SM_PACK_ARRAY_3(float)
void _XCALABLEMP_pack_array_3_DOUBLE			_XCALABLEMP_SM_PACK_ARRAY_3(double)
void _XCALABLEMP_pack_array_3_LONG_DOUBLE		_XCALABLEMP_SM_PACK_ARRAY_3(long double)

// --- dimension 4
#define _XCALABLEMP_SM_PACK_ARRAY_4(_type) \
(_type *buf_addr, _type *src_addr, \
 int src_lower0, int src_upper0, int src_stride0, unsigned long long src_dim_acc0, \
 int src_lower1, int src_upper1, int src_stride1, unsigned long long src_dim_acc1, \
 int src_lower2, int src_upper2, int src_stride2, unsigned long long src_dim_acc2, \
 int src_lower3, int src_upper3, int src_stride3) { \
  _XCALABLEMP_normalize_array_section(&src_lower0, &src_upper0, &src_stride0); \
  _XCALABLEMP_normalize_array_section(&src_lower1, &src_upper1, &src_stride1); \
  _XCALABLEMP_normalize_array_section(&src_lower2, &src_upper2, &src_stride2); \
  _XCALABLEMP_normalize_array_section(&src_lower3, &src_upper3, &src_stride3); \
  for (int l = src_lower0; l <= src_upper0; l += src_stride0) { \
    for (int k = src_lower1; k <= src_upper1; k += src_stride1) { \
      for (int j = src_lower2; j <= src_upper2; j += src_stride2) { \
        _type *addr = src_addr + (l * src_dim_acc0) + (k * src_dim_acc1); + (j * src_dim_acc2); \
        for (int i = src_lower3; i <= src_upper3; i += src_stride3) { \
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
void _XCALABLEMP_pack_array_4_LONGLONG			_XCALABLEMP_SM_PACK_ARRAY_4(long long)
void _XCALABLEMP_pack_array_4_UNSIGNED_LONGLONG		_XCALABLEMP_SM_PACK_ARRAY_4(unsigned long long)
void _XCALABLEMP_pack_array_4_FLOAT			_XCALABLEMP_SM_PACK_ARRAY_4(float)
void _XCALABLEMP_pack_array_4_DOUBLE			_XCALABLEMP_SM_PACK_ARRAY_4(double)
void _XCALABLEMP_pack_array_4_LONG_DOUBLE		_XCALABLEMP_SM_PACK_ARRAY_4(long double)

// --- dimension 5
#define _XCALABLEMP_SM_PACK_ARRAY_5(_type) \
(_type *buf_addr, _type *src_addr, \
 int src_lower0, int src_upper0, int src_stride0, unsigned long long src_dim_acc0, \
 int src_lower1, int src_upper1, int src_stride1, unsigned long long src_dim_acc1, \
 int src_lower2, int src_upper2, int src_stride2, unsigned long long src_dim_acc2, \
 int src_lower3, int src_upper3, int src_stride3, unsigned long long src_dim_acc3, \
 int src_lower4, int src_upper4, int src_stride4) { \
  _XCALABLEMP_normalize_array_section(&src_lower0, &src_upper0, &src_stride0); \
  _XCALABLEMP_normalize_array_section(&src_lower1, &src_upper1, &src_stride1); \
  _XCALABLEMP_normalize_array_section(&src_lower2, &src_upper2, &src_stride2); \
  _XCALABLEMP_normalize_array_section(&src_lower3, &src_upper3, &src_stride3); \
  _XCALABLEMP_normalize_array_section(&src_lower4, &src_upper4, &src_stride4); \
  for (int m = src_lower0; m <= src_upper0; m += src_stride0) { \
    for (int l = src_lower1; l <= src_upper1; l += src_stride1) { \
      for (int k = src_lower2; k <= src_upper2; k += src_stride2) { \
        for (int j = src_lower3; j <= src_upper3; j += src_stride3) { \
          _type *addr = src_addr + (m * src_dim_acc0) + (l * src_dim_acc1); + (k * src_dim_acc2) + (j * src_dim_acc3); \
          for (int i = src_lower4; i <= src_upper4; i += src_stride4) { \
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
void _XCALABLEMP_pack_array_5_LONGLONG			_XCALABLEMP_SM_PACK_ARRAY_5(long long)
void _XCALABLEMP_pack_array_5_UNSIGNED_LONGLONG		_XCALABLEMP_SM_PACK_ARRAY_5(unsigned long long)
void _XCALABLEMP_pack_array_5_FLOAT			_XCALABLEMP_SM_PACK_ARRAY_5(float)
void _XCALABLEMP_pack_array_5_DOUBLE			_XCALABLEMP_SM_PACK_ARRAY_5(double)
void _XCALABLEMP_pack_array_5_LONG_DOUBLE		_XCALABLEMP_SM_PACK_ARRAY_5(long double)

// --- dimension 6
#define _XCALABLEMP_SM_PACK_ARRAY_6(_type) \
(_type *buf_addr, _type *src_addr, \
 int src_lower0, int src_upper0, int src_stride0, unsigned long long src_dim_acc0, \
 int src_lower1, int src_upper1, int src_stride1, unsigned long long src_dim_acc1, \
 int src_lower2, int src_upper2, int src_stride2, unsigned long long src_dim_acc2, \
 int src_lower3, int src_upper3, int src_stride3, unsigned long long src_dim_acc3, \
 int src_lower4, int src_upper4, int src_stride4, unsigned long long src_dim_acc4, \
 int src_lower5, int src_upper5, int src_stride5) { \
  _XCALABLEMP_normalize_array_section(&src_lower0, &src_upper0, &src_stride0); \
  _XCALABLEMP_normalize_array_section(&src_lower1, &src_upper1, &src_stride1); \
  _XCALABLEMP_normalize_array_section(&src_lower2, &src_upper2, &src_stride2); \
  _XCALABLEMP_normalize_array_section(&src_lower3, &src_upper3, &src_stride3); \
  _XCALABLEMP_normalize_array_section(&src_lower4, &src_upper4, &src_stride4); \
  _XCALABLEMP_normalize_array_section(&src_lower5, &src_upper5, &src_stride5); \
  for (int n = src_lower0; n <= src_upper0; n += src_stride0) { \
    for (int m = src_lower1; m <= src_upper1; m += src_stride1) { \
      for (int l = src_lower2; l <= src_upper2; l += src_stride2) { \
        for (int k = src_lower3; k <= src_upper3; k += src_stride3) { \
          for (int j = src_lower4; j <= src_upper4; j += src_stride4) { \
            _type *addr = src_addr + (n * src_dim_acc0) + (m * src_dim_acc1); + (l * src_dim_acc2) + \
                                     (k * src_dim_acc3) + (j * src_dim_acc4); \
            for (int i = src_lower5; i <= src_upper5; i += src_stride5) { \
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
void _XCALABLEMP_pack_array_6_LONGLONG			_XCALABLEMP_SM_PACK_ARRAY_6(long long)
void _XCALABLEMP_pack_array_6_UNSIGNED_LONGLONG		_XCALABLEMP_SM_PACK_ARRAY_6(unsigned long long)
void _XCALABLEMP_pack_array_6_FLOAT			_XCALABLEMP_SM_PACK_ARRAY_6(float)
void _XCALABLEMP_pack_array_6_DOUBLE			_XCALABLEMP_SM_PACK_ARRAY_6(double)
void _XCALABLEMP_pack_array_6_LONG_DOUBLE		_XCALABLEMP_SM_PACK_ARRAY_6(long double)

// --- dimension 7
#define _XCALABLEMP_SM_PACK_ARRAY_7(_type) \
(_type *buf_addr, _type *src_addr, \
 int src_lower0, int src_upper0, int src_stride0, unsigned long long src_dim_acc0, \
 int src_lower1, int src_upper1, int src_stride1, unsigned long long src_dim_acc1, \
 int src_lower2, int src_upper2, int src_stride2, unsigned long long src_dim_acc2, \
 int src_lower3, int src_upper3, int src_stride3, unsigned long long src_dim_acc3, \
 int src_lower4, int src_upper4, int src_stride4, unsigned long long src_dim_acc4, \
 int src_lower5, int src_upper5, int src_stride5, unsigned long long src_dim_acc5, \
 int src_lower6, int src_upper6, int src_stride6) { \
  _XCALABLEMP_normalize_array_section(&src_lower0, &src_upper0, &src_stride0); \
  _XCALABLEMP_normalize_array_section(&src_lower1, &src_upper1, &src_stride1); \
  _XCALABLEMP_normalize_array_section(&src_lower2, &src_upper2, &src_stride2); \
  _XCALABLEMP_normalize_array_section(&src_lower3, &src_upper3, &src_stride3); \
  _XCALABLEMP_normalize_array_section(&src_lower4, &src_upper4, &src_stride4); \
  _XCALABLEMP_normalize_array_section(&src_lower5, &src_upper5, &src_stride5); \
  _XCALABLEMP_normalize_array_section(&src_lower6, &src_upper6, &src_stride6); \
  for (int o = src_lower0; o <= src_upper0; o += src_stride0) { \
    for (int n = src_lower1; n <= src_upper1; n += src_stride1) { \
      for (int m = src_lower2; m <= src_upper2; m += src_stride2) { \
        for (int l = src_lower3; l <= src_upper3; l += src_stride3) { \
          for (int k = src_lower4; k <= src_upper4; k += src_stride4) { \
            for (int j = src_lower5; j <= src_upper5; j += src_stride5) { \
              _type *addr = src_addr + (o * src_dim_acc0) + (n * src_dim_acc1); + (m * src_dim_acc2) + \
                                       (l * src_dim_acc3) + (k * src_dim_acc4); + (j * src_dim_acc5); \
              for (int i = src_lower6; i <= src_upper6; i += src_stride6) { \
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
void _XCALABLEMP_pack_array_7_LONGLONG			_XCALABLEMP_SM_PACK_ARRAY_7(long long)
void _XCALABLEMP_pack_array_7_UNSIGNED_LONGLONG		_XCALABLEMP_SM_PACK_ARRAY_7(unsigned long long)
void _XCALABLEMP_pack_array_7_FLOAT			_XCALABLEMP_SM_PACK_ARRAY_7(float)
void _XCALABLEMP_pack_array_7_DOUBLE			_XCALABLEMP_SM_PACK_ARRAY_7(double)
void _XCALABLEMP_pack_array_7_LONG_DOUBLE		_XCALABLEMP_SM_PACK_ARRAY_7(long double)


// ----- unpack array
// --- dimension 1
#define _XCALABLEMP_SM_UNPACK_ARRAY_1(_type) \
(_type *dst_addr, _type *buf_addr, \
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
void _XCALABLEMP_unpack_array_1_LONGLONG		_XCALABLEMP_SM_UNPACK_ARRAY_1(long long)
void _XCALABLEMP_unpack_array_1_UNSIGNED_LONGLONG	_XCALABLEMP_SM_UNPACK_ARRAY_1(unsigned long long)
void _XCALABLEMP_unpack_array_1_FLOAT			_XCALABLEMP_SM_UNPACK_ARRAY_1(float)
void _XCALABLEMP_unpack_array_1_DOUBLE			_XCALABLEMP_SM_UNPACK_ARRAY_1(double)
void _XCALABLEMP_unpack_array_1_LONG_DOUBLE		_XCALABLEMP_SM_UNPACK_ARRAY_1(long double)

// --- dimension 2
#define _XCALABLEMP_SM_UNPACK_ARRAY_2(_type) \
(_type *dst_addr, _type *buf_addr, \
 int dst_lower0, int dst_upper0, int dst_stride0, unsigned long long dst_dim_acc0, \
 int dst_lower1, int dst_upper1, int dst_stride1) { \
  _XCALABLEMP_normalize_array_section(&dst_lower0, &dst_upper0, &dst_stride0); \
  _XCALABLEMP_normalize_array_section(&dst_lower1, &dst_upper1, &dst_stride1); \
  for (int j = dst_lower0; j <= dst_upper0; j += dst_stride0) { \
    _type *addr = dst_addr + (j * dst_dim_acc0); \
    for (int i = dst_lower1; i <= dst_upper1; i += dst_stride1) { \
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
void _XCALABLEMP_unpack_array_2_LONGLONG		_XCALABLEMP_SM_UNPACK_ARRAY_2(long long)
void _XCALABLEMP_unpack_array_2_UNSIGNED_LONGLONG	_XCALABLEMP_SM_UNPACK_ARRAY_2(unsigned long long)
void _XCALABLEMP_unpack_array_2_FLOAT			_XCALABLEMP_SM_UNPACK_ARRAY_2(float)
void _XCALABLEMP_unpack_array_2_DOUBLE			_XCALABLEMP_SM_UNPACK_ARRAY_2(double)
void _XCALABLEMP_unpack_array_2_LONG_DOUBLE		_XCALABLEMP_SM_UNPACK_ARRAY_2(long double)

// --- dimension 3
#define _XCALABLEMP_SM_UNPACK_ARRAY_3(_type) \
(_type *dst_addr, _type *buf_addr, \
 int dst_lower0, int dst_upper0, int dst_stride0, unsigned long long dst_dim_acc0, \
 int dst_lower1, int dst_upper1, int dst_stride1, unsigned long long dst_dim_acc1, \
 int dst_lower2, int dst_upper2, int dst_stride2) { \
  _XCALABLEMP_normalize_array_section(&dst_lower0, &dst_upper0, &dst_stride0); \
  _XCALABLEMP_normalize_array_section(&dst_lower1, &dst_upper1, &dst_stride1); \
  _XCALABLEMP_normalize_array_section(&dst_lower2, &dst_upper2, &dst_stride2); \
  for (int k = dst_lower0; k <= dst_upper0; k += dst_stride0) { \
    for (int j = dst_lower1; j <= dst_upper1; j += dst_stride1) { \
      _type *addr = dst_addr + (k * dst_dim_acc0) + (j * dst_dim_acc1); \
      for (int i = dst_lower2; i <= dst_upper2; i += dst_stride2) { \
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
void _XCALABLEMP_unpack_array_3_LONGLONG		_XCALABLEMP_SM_UNPACK_ARRAY_3(long long)
void _XCALABLEMP_unpack_array_3_UNSIGNED_LONGLONG	_XCALABLEMP_SM_UNPACK_ARRAY_3(unsigned long long)
void _XCALABLEMP_unpack_array_3_FLOAT			_XCALABLEMP_SM_UNPACK_ARRAY_3(float)
void _XCALABLEMP_unpack_array_3_DOUBLE			_XCALABLEMP_SM_UNPACK_ARRAY_3(double)
void _XCALABLEMP_unpack_array_3_LONG_DOUBLE		_XCALABLEMP_SM_UNPACK_ARRAY_3(long double)

// --- dimension 4
#define _XCALABLEMP_SM_UNPACK_ARRAY_4(_type) \
(_type *dst_addr, _type *buf_addr, \
 int dst_lower0, int dst_upper0, int dst_stride0, unsigned long long dst_dim_acc0, \
 int dst_lower1, int dst_upper1, int dst_stride1, unsigned long long dst_dim_acc1, \
 int dst_lower2, int dst_upper2, int dst_stride2, unsigned long long dst_dim_acc2, \
 int dst_lower3, int dst_upper3, int dst_stride3) { \
  _XCALABLEMP_normalize_array_section(&dst_lower0, &dst_upper0, &dst_stride0); \
  _XCALABLEMP_normalize_array_section(&dst_lower1, &dst_upper1, &dst_stride1); \
  _XCALABLEMP_normalize_array_section(&dst_lower2, &dst_upper2, &dst_stride2); \
  _XCALABLEMP_normalize_array_section(&dst_lower3, &dst_upper3, &dst_stride3); \
  for (int l = dst_lower0; l <= dst_upper0; l += dst_stride0) { \
    for (int k = dst_lower1; k <= dst_upper1; k += dst_stride1) { \
      for (int j = dst_lower2; j <= dst_upper2; j += dst_stride2) { \
        _type *addr = dst_addr + (l * dst_dim_acc0) + (k * dst_dim_acc1) + (j * dst_dim_acc2); \
        for (int i = dst_lower3; i <= dst_upper3; i += dst_stride3) { \
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
void _XCALABLEMP_unpack_array_4_LONGLONG		_XCALABLEMP_SM_UNPACK_ARRAY_4(long long)
void _XCALABLEMP_unpack_array_4_UNSIGNED_LONGLONG	_XCALABLEMP_SM_UNPACK_ARRAY_4(unsigned long long)
void _XCALABLEMP_unpack_array_4_FLOAT			_XCALABLEMP_SM_UNPACK_ARRAY_4(float)
void _XCALABLEMP_unpack_array_4_DOUBLE			_XCALABLEMP_SM_UNPACK_ARRAY_4(double)
void _XCALABLEMP_unpack_array_4_LONG_DOUBLE		_XCALABLEMP_SM_UNPACK_ARRAY_4(long double)

// --- dimension 5
#define _XCALABLEMP_SM_UNPACK_ARRAY_5(_type) \
(_type *dst_addr, _type *buf_addr, \
 int dst_lower0, int dst_upper0, int dst_stride0, unsigned long long dst_dim_acc0, \
 int dst_lower1, int dst_upper1, int dst_stride1, unsigned long long dst_dim_acc1, \
 int dst_lower2, int dst_upper2, int dst_stride2, unsigned long long dst_dim_acc2, \
 int dst_lower3, int dst_upper3, int dst_stride3, unsigned long long dst_dim_acc3, \
 int dst_lower4, int dst_upper4, int dst_stride4) { \
  _XCALABLEMP_normalize_array_section(&dst_lower0, &dst_upper0, &dst_stride0); \
  _XCALABLEMP_normalize_array_section(&dst_lower1, &dst_upper1, &dst_stride1); \
  _XCALABLEMP_normalize_array_section(&dst_lower2, &dst_upper2, &dst_stride2); \
  _XCALABLEMP_normalize_array_section(&dst_lower3, &dst_upper3, &dst_stride3); \
  _XCALABLEMP_normalize_array_section(&dst_lower4, &dst_upper4, &dst_stride4); \
  for (int m = dst_lower0; m <= dst_upper0; m += dst_stride0) { \
    for (int l = dst_lower1; l <= dst_upper1; l += dst_stride1) { \
      for (int k = dst_lower2; k <= dst_upper2; k += dst_stride2) { \
        for (int j = dst_lower3; j <= dst_upper3; j += dst_stride3) { \
          _type *addr = dst_addr + (m * dst_dim_acc0) + (l * dst_dim_acc1) + (k * dst_dim_acc2) + (j * dst_dim_acc3); \
          for (int i = dst_lower4; i <= dst_upper4; i += dst_stride4) { \
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
void _XCALABLEMP_unpack_array_5_LONGLONG		_XCALABLEMP_SM_UNPACK_ARRAY_5(long long)
void _XCALABLEMP_unpack_array_5_UNSIGNED_LONGLONG	_XCALABLEMP_SM_UNPACK_ARRAY_5(unsigned long long)
void _XCALABLEMP_unpack_array_5_FLOAT			_XCALABLEMP_SM_UNPACK_ARRAY_5(float)
void _XCALABLEMP_unpack_array_5_DOUBLE			_XCALABLEMP_SM_UNPACK_ARRAY_5(double)
void _XCALABLEMP_unpack_array_5_LONG_DOUBLE		_XCALABLEMP_SM_UNPACK_ARRAY_5(long double)

// --- dimension 6
#define _XCALABLEMP_SM_UNPACK_ARRAY_6(_type) \
(_type *dst_addr, _type *buf_addr, \
 int dst_lower0, int dst_upper0, int dst_stride0, unsigned long long dst_dim_acc0, \
 int dst_lower1, int dst_upper1, int dst_stride1, unsigned long long dst_dim_acc1, \
 int dst_lower2, int dst_upper2, int dst_stride2, unsigned long long dst_dim_acc2, \
 int dst_lower3, int dst_upper3, int dst_stride3, unsigned long long dst_dim_acc3, \
 int dst_lower4, int dst_upper4, int dst_stride4, unsigned long long dst_dim_acc4, \
 int dst_lower5, int dst_upper5, int dst_stride5) { \
  _XCALABLEMP_normalize_array_section(&dst_lower0, &dst_upper0, &dst_stride0); \
  _XCALABLEMP_normalize_array_section(&dst_lower1, &dst_upper1, &dst_stride1); \
  _XCALABLEMP_normalize_array_section(&dst_lower2, &dst_upper2, &dst_stride2); \
  _XCALABLEMP_normalize_array_section(&dst_lower3, &dst_upper3, &dst_stride3); \
  _XCALABLEMP_normalize_array_section(&dst_lower4, &dst_upper4, &dst_stride4); \
  _XCALABLEMP_normalize_array_section(&dst_lower5, &dst_upper5, &dst_stride5); \
  for (int n = dst_lower0; n <= dst_upper0; n += dst_stride0) { \
    for (int m = dst_lower1; m <= dst_upper1; m += dst_stride1) { \
      for (int l = dst_lower2; l <= dst_upper2; l += dst_stride2) { \
        for (int k = dst_lower3; k <= dst_upper3; k += dst_stride3) { \
          for (int j = dst_lower4; j <= dst_upper4; j += dst_stride4) { \
            _type *addr = dst_addr + (n * dst_dim_acc0) + (m * dst_dim_acc1) + (l * dst_dim_acc2) + \
                                     (k * dst_dim_acc3) + (j * dst_dim_acc4); \
            for (int i = dst_lower5; i <= dst_upper5; i += dst_stride5) { \
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
void _XCALABLEMP_unpack_array_6_LONGLONG		_XCALABLEMP_SM_UNPACK_ARRAY_6(long long)
void _XCALABLEMP_unpack_array_6_UNSIGNED_LONGLONG	_XCALABLEMP_SM_UNPACK_ARRAY_6(unsigned long long)
void _XCALABLEMP_unpack_array_6_FLOAT			_XCALABLEMP_SM_UNPACK_ARRAY_6(float)
void _XCALABLEMP_unpack_array_6_DOUBLE			_XCALABLEMP_SM_UNPACK_ARRAY_6(double)
void _XCALABLEMP_unpack_array_6_LONG_DOUBLE		_XCALABLEMP_SM_UNPACK_ARRAY_6(long double)

// --- dimension 7
#define _XCALABLEMP_SM_UNPACK_ARRAY_7(_type) \
(_type *dst_addr, _type *buf_addr, \
 int dst_lower0, int dst_upper0, int dst_stride0, unsigned long long dst_dim_acc0, \
 int dst_lower1, int dst_upper1, int dst_stride1, unsigned long long dst_dim_acc1, \
 int dst_lower2, int dst_upper2, int dst_stride2, unsigned long long dst_dim_acc2, \
 int dst_lower3, int dst_upper3, int dst_stride3, unsigned long long dst_dim_acc3, \
 int dst_lower4, int dst_upper4, int dst_stride4, unsigned long long dst_dim_acc4, \
 int dst_lower5, int dst_upper5, int dst_stride5, unsigned long long dst_dim_acc5, \
 int dst_lower6, int dst_upper6, int dst_stride6) { \
  _XCALABLEMP_normalize_array_section(&dst_lower0, &dst_upper0, &dst_stride0); \
  _XCALABLEMP_normalize_array_section(&dst_lower1, &dst_upper1, &dst_stride1); \
  _XCALABLEMP_normalize_array_section(&dst_lower2, &dst_upper2, &dst_stride2); \
  _XCALABLEMP_normalize_array_section(&dst_lower3, &dst_upper3, &dst_stride3); \
  _XCALABLEMP_normalize_array_section(&dst_lower4, &dst_upper4, &dst_stride4); \
  _XCALABLEMP_normalize_array_section(&dst_lower5, &dst_upper5, &dst_stride5); \
  _XCALABLEMP_normalize_array_section(&dst_lower6, &dst_upper6, &dst_stride6); \
  for (int o = dst_lower0; o <= dst_upper0; o += dst_stride0) { \
    for (int n = dst_lower1; n <= dst_upper1; n += dst_stride1) { \
      for (int m = dst_lower2; m <= dst_upper2; m += dst_stride2) { \
        for (int l = dst_lower3; l <= dst_upper3; l += dst_stride3) { \
          for (int k = dst_lower4; k <= dst_upper4; k += dst_stride4) { \
            for (int j = dst_lower5; j <= dst_upper5; j += dst_stride5) { \
              _type *addr = dst_addr + (o * dst_dim_acc0) + (n * dst_dim_acc1) + (m * dst_dim_acc2) + \
                                       (l * dst_dim_acc3) + (k * dst_dim_acc4) + (j * dst_dim_acc5); \
              for (int i = dst_lower6; i <= dst_upper6; i += dst_stride6) { \
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
void _XCALABLEMP_unpack_array_7_LONGLONG		_XCALABLEMP_SM_UNPACK_ARRAY_7(long long)
void _XCALABLEMP_unpack_array_7_UNSIGNED_LONGLONG	_XCALABLEMP_SM_UNPACK_ARRAY_7(unsigned long long)
void _XCALABLEMP_unpack_array_7_FLOAT			_XCALABLEMP_SM_UNPACK_ARRAY_7(float)
void _XCALABLEMP_unpack_array_7_DOUBLE			_XCALABLEMP_SM_UNPACK_ARRAY_7(double)
void _XCALABLEMP_unpack_array_7_LONG_DOUBLE		_XCALABLEMP_SM_UNPACK_ARRAY_7(long double)


// pack shadow
void _XCALABLEMP_pack_shadow_buffer(void *buffer, void *src,
                                    int array_type, int array_dim, int *l, int *u, int *s, unsigned long long *d) {
  switch (array_type) {
 // case _XCALABLEMP_N_TYPE_BOOL:
 //   _XCALABLEMP_fatal("unknown data type for reflect");
    case _XCALABLEMP_N_TYPE_CHAR:
      _XCALABLEMP_fatal("unknown data type for reflect");
    case _XCALABLEMP_N_TYPE_UNSIGNED_CHAR:
      _XCALABLEMP_fatal("unknown data type for reflect");
    case _XCALABLEMP_N_TYPE_SHORT:
      _XCALABLEMP_fatal("unknown data type for reflect");
    case _XCALABLEMP_N_TYPE_UNSIGNED_SHORT:
      _XCALABLEMP_fatal("unknown data type for reflect");
    case _XCALABLEMP_N_TYPE_INT: {
        switch (array_dim) {
          case 1: _XCALABLEMP_pack_array_1_INT(buffer, src, l[0], u[0], s[0]); break;
          case 2: _XCALABLEMP_pack_array_2_INT(buffer, src, l[0], u[0], s[0], d[0],
                                                            l[1], u[1], s[1]); break;
          case 3: _XCALABLEMP_pack_array_3_INT(buffer, src, l[0], u[0], s[0], d[0],
                                                            l[1], u[1], s[1], d[1],
                                                            l[2], u[2], s[2]); break;
          case 4: _XCALABLEMP_pack_array_4_INT(buffer, src, l[0], u[0], s[0], d[0],
                                                            l[1], u[1], s[1], d[1],
                                                            l[2], u[2], s[2], d[2],
                                                            l[3], u[3], s[3]); break;
          case 5: _XCALABLEMP_pack_array_5_INT(buffer, src, l[0], u[0], s[0], d[0],
                                                            l[1], u[1], s[1], d[1],
                                                            l[2], u[2], s[2], d[2],
                                                            l[3], u[3], s[3], d[3],
                                                            l[4], u[4], s[4]); break;
          case 6: _XCALABLEMP_pack_array_6_INT(buffer, src, l[0], u[0], s[0], d[0],
                                                            l[1], u[1], s[1], d[1],
                                                            l[2], u[2], s[2], d[2],
                                                            l[3], u[3], s[3], d[3],
                                                            l[4], u[4], s[4], d[4],
                                                            l[5], u[5], s[5]); break;
          case 7: _XCALABLEMP_pack_array_7_INT(buffer, src, l[0], u[0], s[0], d[0],
                                                            l[1], u[1], s[1], d[1],
                                                            l[2], u[2], s[2], d[2],
                                                            l[3], u[3], s[3], d[3],
                                                            l[4], u[4], s[4], d[4],
                                                            l[5], u[5], s[5], d[5],
                                                            l[6], u[6], s[6]); break;
          default: _XCALABLEMP_fatal("wrong array dimension");
        }
      } break;
    case _XCALABLEMP_N_TYPE_UNSIGNED_INT:
      _XCALABLEMP_fatal("unknown data type for reflect");
    case _XCALABLEMP_N_TYPE_LONG:
      _XCALABLEMP_fatal("unknown data type for reflect");
    case _XCALABLEMP_N_TYPE_UNSIGNED_LONG:
      _XCALABLEMP_fatal("unknown data type for reflect");
    case _XCALABLEMP_N_TYPE_LONGLONG:
      _XCALABLEMP_fatal("unknown data type for reflect");
    case _XCALABLEMP_N_TYPE_UNSIGNED_LONGLONG:
      _XCALABLEMP_fatal("unknown data type for reflect");
    case _XCALABLEMP_N_TYPE_FLOAT: {
        switch (array_dim) {
          case 1: _XCALABLEMP_pack_array_1_FLOAT(buffer, src, l[0], u[0], s[0]); break;
          case 2: _XCALABLEMP_pack_array_2_FLOAT(buffer, src, l[0], u[0], s[0], d[0],
                                                              l[1], u[1], s[1]); break;
          case 3: _XCALABLEMP_pack_array_3_FLOAT(buffer, src, l[0], u[0], s[0], d[0],
                                                              l[1], u[1], s[1], d[1],
                                                              l[2], u[2], s[2]); break;
          case 4: _XCALABLEMP_pack_array_4_FLOAT(buffer, src, l[0], u[0], s[0], d[0],
                                                              l[1], u[1], s[1], d[1],
                                                              l[2], u[2], s[2], d[2],
                                                              l[3], u[3], s[3]); break;
          case 5: _XCALABLEMP_pack_array_5_FLOAT(buffer, src, l[0], u[0], s[0], d[0],
                                                              l[1], u[1], s[1], d[1],
                                                              l[2], u[2], s[2], d[2],
                                                              l[3], u[3], s[3], d[3],
                                                              l[4], u[4], s[4]); break;
          case 6: _XCALABLEMP_pack_array_6_FLOAT(buffer, src, l[0], u[0], s[0], d[0],
                                                              l[1], u[1], s[1], d[1],
                                                              l[2], u[2], s[2], d[2],
                                                              l[3], u[3], s[3], d[3],
                                                              l[4], u[4], s[4], d[4],
                                                              l[5], u[5], s[5]); break;
          case 7: _XCALABLEMP_pack_array_7_FLOAT(buffer, src, l[0], u[0], s[0], d[0],
                                                              l[1], u[1], s[1], d[1],
                                                              l[2], u[2], s[2], d[2],
                                                              l[3], u[3], s[3], d[3],
                                                              l[4], u[4], s[4], d[4],
                                                              l[5], u[5], s[5], d[5],
                                                              l[6], u[6], s[6]); break;
          default: _XCALABLEMP_fatal("wrong array dimension");
        }
      } break;
    case _XCALABLEMP_N_TYPE_DOUBLE: {
        switch (array_dim) {
          case 1: _XCALABLEMP_pack_array_1_DOUBLE(buffer, src, l[0], u[0], s[0]); break;
          case 2: _XCALABLEMP_pack_array_2_DOUBLE(buffer, src, l[0], u[0], s[0], d[0],
                                                               l[1], u[1], s[1]); break;
          case 3: _XCALABLEMP_pack_array_3_DOUBLE(buffer, src, l[0], u[0], s[0], d[0],
                                                               l[1], u[1], s[1], d[1],
                                                               l[2], u[2], s[2]); break;
          case 4: _XCALABLEMP_pack_array_4_DOUBLE(buffer, src, l[0], u[0], s[0], d[0],
                                                               l[1], u[1], s[1], d[1],
                                                               l[2], u[2], s[2], d[2],
                                                               l[3], u[3], s[3]); break;
          case 5: _XCALABLEMP_pack_array_5_DOUBLE(buffer, src, l[0], u[0], s[0], d[0],
                                                               l[1], u[1], s[1], d[1],
                                                               l[2], u[2], s[2], d[2],
                                                               l[3], u[3], s[3], d[3],
                                                               l[4], u[4], s[4]); break;
          case 6: _XCALABLEMP_pack_array_6_DOUBLE(buffer, src, l[0], u[0], s[0], d[0],
                                                               l[1], u[1], s[1], d[1],
                                                               l[2], u[2], s[2], d[2],
                                                               l[3], u[3], s[3], d[3],
                                                               l[4], u[4], s[4], d[4],
                                                               l[5], u[5], s[5]); break;
          case 7: _XCALABLEMP_pack_array_7_DOUBLE(buffer, src, l[0], u[0], s[0], d[0],
                                                               l[1], u[1], s[1], d[1],
                                                               l[2], u[2], s[2], d[2],
                                                               l[3], u[3], s[3], d[3],
                                                               l[4], u[4], s[4], d[4],
                                                               l[5], u[5], s[5], d[5],
                                                               l[6], u[6], s[6]); break;
          default: _XCALABLEMP_fatal("wrong array dimension");
        }
      } break;
    case _XCALABLEMP_N_TYPE_LONG_DOUBLE:
      _XCALABLEMP_fatal("unknown data type for reflect");
 // case _XCALABLEMP_N_TYPE_FLOAT_IMAGINARY:
 //   _XCALABLEMP_fatal("unknown data type for reflect");
 // case _XCALABLEMP_N_TYPE_DOUBLE_IMAGINARY:
 //   _XCALABLEMP_fatal("unknown data type for reflect");
 // case _XCALABLEMP_N_TYPE_LONG_DOUBLE_IMAGINARY:
 //   _XCALABLEMP_fatal("unknown data type for reflect");
 // case _XCALABLEMP_N_TYPE_FLOAT_COMPLEX:
 //   _XCALABLEMP_fatal("unknown data type for reflect");
 // case _XCALABLEMP_N_TYPE_DOUBLE_COMPLEX:
 //   _XCALABLEMP_fatal("unknown data type for reflect");
 // case _XCALABLEMP_N_TYPE_LONG_DOUBLE_COMPLEX:
    default:
      _XCALABLEMP_fatal("unknown data type for reflect");
  }
}

// unpack shadow
void _XCALABLEMP_unpack_shadow_buffer(void *dst, void *buffer,
                                      int array_type, int array_dim, int *l, int *u, int *s, unsigned long long *d) {
  switch (array_type) {
 // case _XCALABLEMP_N_TYPE_BOOL:
 //   _XCALABLEMP_fatal("unknown data type for reflect");
    case _XCALABLEMP_N_TYPE_CHAR:
      _XCALABLEMP_fatal("unknown data type for reflect");
    case _XCALABLEMP_N_TYPE_UNSIGNED_CHAR:
      _XCALABLEMP_fatal("unknown data type for reflect");
    case _XCALABLEMP_N_TYPE_SHORT:
      _XCALABLEMP_fatal("unknown data type for reflect");
    case _XCALABLEMP_N_TYPE_UNSIGNED_SHORT:
      _XCALABLEMP_fatal("unknown data type for reflect");
    case _XCALABLEMP_N_TYPE_INT: {
        switch (array_dim) {
          case 1: _XCALABLEMP_unpack_array_1_INT(dst, buffer, l[0], u[0], s[0]); break;
          case 2: _XCALABLEMP_unpack_array_2_INT(dst, buffer, l[0], u[0], s[0], d[0],
                                                              l[1], u[1], s[1]); break;
          case 3: _XCALABLEMP_unpack_array_3_INT(dst, buffer, l[0], u[0], s[0], d[0],
                                                              l[1], u[1], s[1], d[1],
                                                              l[2], u[2], s[2]); break;
          case 4: _XCALABLEMP_unpack_array_4_INT(dst, buffer, l[0], u[0], s[0], d[0],
                                                              l[1], u[1], s[1], d[1],
                                                              l[2], u[2], s[2], d[2],
                                                              l[3], u[3], s[3]); break;
          case 5: _XCALABLEMP_unpack_array_5_INT(dst, buffer, l[0], u[0], s[0], d[0],
                                                              l[1], u[1], s[1], d[1],
                                                              l[2], u[2], s[2], d[2],
                                                              l[3], u[3], s[3], d[3],
                                                              l[4], u[4], s[4]); break;
          case 6: _XCALABLEMP_unpack_array_6_INT(dst, buffer, l[0], u[0], s[0], d[0],
                                                              l[1], u[1], s[1], d[1],
                                                              l[2], u[2], s[2], d[2],
                                                              l[3], u[3], s[3], d[3],
                                                              l[4], u[4], s[4], d[4],
                                                              l[5], u[5], s[5]); break;
          case 7: _XCALABLEMP_unpack_array_7_INT(dst, buffer, l[0], u[0], s[0], d[0],
                                                              l[1], u[1], s[1], d[1],
                                                              l[2], u[2], s[2], d[2],
                                                              l[3], u[3], s[3], d[3],
                                                              l[4], u[4], s[4], d[4],
                                                              l[5], u[5], s[5], d[5],
                                                              l[6], u[6], s[6]); break;
          default: _XCALABLEMP_fatal("wrong array dimension");
        }
      } break;
    case _XCALABLEMP_N_TYPE_UNSIGNED_INT:
      _XCALABLEMP_fatal("unknown data type for reflect");
    case _XCALABLEMP_N_TYPE_LONG:
      _XCALABLEMP_fatal("unknown data type for reflect");
    case _XCALABLEMP_N_TYPE_UNSIGNED_LONG:
      _XCALABLEMP_fatal("unknown data type for reflect");
    case _XCALABLEMP_N_TYPE_LONGLONG:
      _XCALABLEMP_fatal("unknown data type for reflect");
    case _XCALABLEMP_N_TYPE_UNSIGNED_LONGLONG:
      _XCALABLEMP_fatal("unknown data type for reflect");
    case _XCALABLEMP_N_TYPE_FLOAT: {
        switch (array_dim) {
          case 1: _XCALABLEMP_unpack_array_1_FLOAT(dst, buffer, l[0], u[0], s[0]); break;
          case 2: _XCALABLEMP_unpack_array_2_FLOAT(dst, buffer, l[0], u[0], s[0], d[0],
                                                                l[1], u[1], s[1]); break;
          case 3: _XCALABLEMP_unpack_array_3_FLOAT(dst, buffer, l[0], u[0], s[0], d[0],
                                                                l[1], u[1], s[1], d[1],
                                                                l[2], u[2], s[2]); break;
          case 4: _XCALABLEMP_unpack_array_4_FLOAT(dst, buffer, l[0], u[0], s[0], d[0],
                                                                l[1], u[1], s[1], d[1],
                                                                l[2], u[2], s[2], d[2],
                                                                l[3], u[3], s[3]); break;
          case 5: _XCALABLEMP_unpack_array_5_FLOAT(dst, buffer, l[0], u[0], s[0], d[0],
                                                                l[1], u[1], s[1], d[1],
                                                                l[2], u[2], s[2], d[2],
                                                                l[3], u[3], s[3], d[3],
                                                                l[4], u[4], s[4]); break;
          case 6: _XCALABLEMP_unpack_array_6_FLOAT(dst, buffer, l[0], u[0], s[0], d[0],
                                                                l[1], u[1], s[1], d[1],
                                                                l[2], u[2], s[2], d[2],
                                                                l[3], u[3], s[3], d[3],
                                                                l[4], u[4], s[4], d[4],
                                                                l[5], u[5], s[5]); break;
          case 7: _XCALABLEMP_unpack_array_7_FLOAT(dst, buffer, l[0], u[0], s[0], d[0],
                                                                l[1], u[1], s[1], d[1],
                                                                l[2], u[2], s[2], d[2],
                                                                l[3], u[3], s[3], d[3],
                                                                l[4], u[4], s[4], d[4],
                                                                l[5], u[5], s[5], d[5],
                                                                l[6], u[6], s[6]); break;
          default: _XCALABLEMP_fatal("wrong array dimension");
        }
      } break;
    case _XCALABLEMP_N_TYPE_DOUBLE: {
        switch (array_dim) {
          case 1: _XCALABLEMP_unpack_array_1_DOUBLE(dst, buffer, l[0], u[0], s[0]); break;
          case 2: _XCALABLEMP_unpack_array_2_DOUBLE(dst, buffer, l[0], u[0], s[0], d[0],
                                                                 l[1], u[1], s[1]); break;
          case 3: _XCALABLEMP_unpack_array_3_DOUBLE(dst, buffer, l[0], u[0], s[0], d[0],
                                                                 l[1], u[1], s[1], d[1],
                                                                 l[2], u[2], s[2]); break;
          case 4: _XCALABLEMP_unpack_array_4_DOUBLE(dst, buffer, l[0], u[0], s[0], d[0],
                                                                 l[1], u[1], s[1], d[1],
                                                                 l[2], u[2], s[2], d[2],
                                                                 l[3], u[3], s[3]); break;
          case 5: _XCALABLEMP_unpack_array_5_DOUBLE(dst, buffer, l[0], u[0], s[0], d[0],
                                                                 l[1], u[1], s[1], d[1],
                                                                 l[2], u[2], s[2], d[2],
                                                                 l[3], u[3], s[3], d[3],
                                                                 l[4], u[4], s[4]); break;
          case 6: _XCALABLEMP_unpack_array_6_DOUBLE(dst, buffer, l[0], u[0], s[0], d[0],
                                                                 l[1], u[1], s[1], d[1],
                                                                 l[2], u[2], s[2], d[2],
                                                                 l[3], u[3], s[3], d[3],
                                                                 l[4], u[4], s[4], d[4],
                                                                 l[5], u[5], s[5]); break;
          case 7: _XCALABLEMP_unpack_array_7_DOUBLE(dst, buffer, l[0], u[0], s[0], d[0],
                                                                 l[1], u[1], s[1], d[1],
                                                                 l[2], u[2], s[2], d[2],
                                                                 l[3], u[3], s[3], d[3],
                                                                 l[4], u[4], s[4], d[4],
                                                                 l[5], u[5], s[5], d[5],
                                                                 l[6], u[6], s[6]); break;
          default: _XCALABLEMP_fatal("wrong array dimension");
        }
      } break;
    case _XCALABLEMP_N_TYPE_LONG_DOUBLE:
      _XCALABLEMP_fatal("unknown data type for reflect");
 // case _XCALABLEMP_N_TYPE_FLOAT_IMAGINARY:
 //   _XCALABLEMP_fatal("unknown data type for reflect");
 // case _XCALABLEMP_N_TYPE_DOUBLE_IMAGINARY:
 //   _XCALABLEMP_fatal("unknown data type for reflect");
 // case _XCALABLEMP_N_TYPE_LONG_DOUBLE_IMAGINARY:
 //   _XCALABLEMP_fatal("unknown data type for reflect");
 // case _XCALABLEMP_N_TYPE_FLOAT_COMPLEX:
 //   _XCALABLEMP_fatal("unknown data type for reflect");
 // case _XCALABLEMP_N_TYPE_DOUBLE_COMPLEX:
 //   _XCALABLEMP_fatal("unknown data type for reflect");
 // case _XCALABLEMP_N_TYPE_LONG_DOUBLE_COMPLEX:
 //   _XCALABLEMP_fatal("unknown data type for reflect");
    default:
      _XCALABLEMP_fatal("unknown data type for reflect");
  }
}
