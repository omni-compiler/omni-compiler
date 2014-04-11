/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include "xmpf_internal.h"

// ----- pack array

// --- dimension 1
#define _XMP_SM_PACK_ARRAY_1(_type) \
(_type *buf_addr, _type *src_addr, \
 int src_lower, int src_upper, int src_stride) { \
  for (int i = src_lower; i <= src_upper; i += src_stride) { \
    *buf_addr = src_addr[i]; \
    buf_addr++; \
  } \
}

static void _XMP_pack_array_1_BOOL		_XMP_SM_PACK_ARRAY_1(_Bool)
static void _XMP_pack_array_1_CHAR		_XMP_SM_PACK_ARRAY_1(char)
static void _XMP_pack_array_1_UNSIGNED_CHAR	_XMP_SM_PACK_ARRAY_1(unsigned char)
static void _XMP_pack_array_1_SHORT		_XMP_SM_PACK_ARRAY_1(short)
static void _XMP_pack_array_1_UNSIGNED_SHORT	_XMP_SM_PACK_ARRAY_1(unsigned short)
static void _XMP_pack_array_1_INT		_XMP_SM_PACK_ARRAY_1(int)
static void _XMP_pack_array_1_UNSIGNED_INT	_XMP_SM_PACK_ARRAY_1(unsigned int)
static void _XMP_pack_array_1_LONG		_XMP_SM_PACK_ARRAY_1(long)
static void _XMP_pack_array_1_UNSIGNED_LONG	_XMP_SM_PACK_ARRAY_1(unsigned long)
static void _XMP_pack_array_1_LONGLONG		_XMP_SM_PACK_ARRAY_1(long long)
static void _XMP_pack_array_1_UNSIGNED_LONGLONG	_XMP_SM_PACK_ARRAY_1(unsigned long long)
static void _XMP_pack_array_1_FLOAT		_XMP_SM_PACK_ARRAY_1(float)
static void _XMP_pack_array_1_DOUBLE		_XMP_SM_PACK_ARRAY_1(double)
static void _XMP_pack_array_1_LONG_DOUBLE	_XMP_SM_PACK_ARRAY_1(long double)
static void _XMP_pack_array_1_GENERAL(void *buf_addr, void *src_addr, size_t type_size,
				      int src_lower, int src_upper, int src_stride) {
  if (src_stride == 1) {
    memcpy(buf_addr, (char *)src_addr + (type_size * src_lower),
           type_size * _XMP_M_COUNT_TRIPLETi(src_lower, src_upper, src_stride));
  }
  else {
    void *addr = (char *)src_addr + (type_size * src_lower);
    int byte_stride = type_size * src_stride;
    for (int i = src_lower; i <= src_upper; i += src_stride) {
      memcpy(buf_addr, addr, type_size);
      buf_addr = (char *)buf_addr + type_size;
      addr = (char *)addr + byte_stride;
    }
  }
}

// --- dimension 2
#define _XMP_SM_PACK_ARRAY_2(_type) \
(_type *buf_addr, _type *src_addr, \
 int *l, int *u, int *s, unsigned long long *d) { \
  int src_lower0 = l[0]; int src_upper0 = u[0]; int src_stride0 = s[0]; \
  int src_lower1 = l[1]; int src_upper1 = u[1]; int src_stride1 = s[1]; unsigned long long src_dim_acc1 = d[1]; \
  for (int j = src_lower1; j <= src_upper1; j += src_stride1) { \
    _type *addr = src_addr + (j * src_dim_acc1); \
    for (int i = src_lower0; i <= src_upper0; i += src_stride0) { \
      *buf_addr = addr[i]; \
      buf_addr++; \
    } \
  } \
}

static void _XMP_pack_array_2_BOOL		_XMP_SM_PACK_ARRAY_2(_Bool)
static void _XMP_pack_array_2_CHAR		_XMP_SM_PACK_ARRAY_2(char)
static void _XMP_pack_array_2_UNSIGNED_CHAR	_XMP_SM_PACK_ARRAY_2(unsigned char)
static void _XMP_pack_array_2_SHORT		_XMP_SM_PACK_ARRAY_2(short)
static void _XMP_pack_array_2_UNSIGNED_SHORT	_XMP_SM_PACK_ARRAY_2(unsigned short)
static void _XMP_pack_array_2_INT		_XMP_SM_PACK_ARRAY_2(int)
static void _XMP_pack_array_2_UNSIGNED_INT	_XMP_SM_PACK_ARRAY_2(unsigned int)
static void _XMP_pack_array_2_LONG		_XMP_SM_PACK_ARRAY_2(long)
static void _XMP_pack_array_2_UNSIGNED_LONG	_XMP_SM_PACK_ARRAY_2(unsigned long)
static void _XMP_pack_array_2_LONGLONG		_XMP_SM_PACK_ARRAY_2(long long)
static void _XMP_pack_array_2_UNSIGNED_LONGLONG	_XMP_SM_PACK_ARRAY_2(unsigned long long)
static void _XMP_pack_array_2_FLOAT		_XMP_SM_PACK_ARRAY_2(float)
static void _XMP_pack_array_2_DOUBLE		_XMP_SM_PACK_ARRAY_2(double)
static void _XMP_pack_array_2_LONG_DOUBLE	_XMP_SM_PACK_ARRAY_2(long double)

static void _XMP_pack_array_2_GENERAL(void *buf_addr, void *src_addr, size_t type_size,
				      int *l, int *u, int *s, unsigned long long *d) {
  int src_lower0 = l[0]; int src_upper0 = u[0]; int src_stride0 = s[0];
  int src_lower1 = l[1]; int src_upper1 = u[1]; int src_stride1 = s[1]; unsigned long long src_dim_acc1 = d[1];
  for (int j = src_lower1; j <= src_upper1; j += src_stride1) {
    void *addr = (char *)src_addr + (type_size * ((j * src_dim_acc1) + src_lower0));
    if (src_stride0 == 1) {
      size_t n = type_size * _XMP_M_COUNT_TRIPLETi(src_lower0, src_upper0, src_stride0);
      memcpy(buf_addr, addr, n);
      buf_addr = (char *)buf_addr + n;
    }
    else {
      int byte_stride = type_size * src_stride0;
      for (int i = src_lower0; i <= src_upper0; i += src_stride0) {
        memcpy(buf_addr, addr, type_size);
        buf_addr = (char *)buf_addr + type_size;
        addr = (char *)addr + byte_stride;
      }
    }
  }
}

// --- dimension 3
#define _XMP_SM_PACK_ARRAY_3(_type) \
(_type *buf_addr, _type *src_addr, \
 int *l, int *u, int *s, unsigned long long *d) { \
  int src_lower0 = l[0]; int src_upper0 = u[0]; int src_stride0 = s[0]; \
  int src_lower1 = l[1]; int src_upper1 = u[1]; int src_stride1 = s[1]; unsigned long long src_dim_acc1 = d[1]; \
  int src_lower2 = l[2]; int src_upper2 = u[2]; int src_stride2 = s[2]; unsigned long long src_dim_acc2 = d[2]; \
  for (int k = src_lower2; k <= src_upper2; k += src_stride2) { \
  for (int j = src_lower1; j <= src_upper1; j += src_stride1) { \
    _type *addr = src_addr + (k * src_dim_acc2) + (j * src_dim_acc1);	\
    for (int i = src_lower0; i <= src_upper0; i += src_stride0) {	\
      *buf_addr = addr[i];						\
      buf_addr++;							\
    }									\
  }									\
  } \
}

static void _XMP_pack_array_3_BOOL		_XMP_SM_PACK_ARRAY_3(_Bool)
static void _XMP_pack_array_3_CHAR		_XMP_SM_PACK_ARRAY_3(char)
static void _XMP_pack_array_3_UNSIGNED_CHAR	_XMP_SM_PACK_ARRAY_3(unsigned char)
static void _XMP_pack_array_3_SHORT		_XMP_SM_PACK_ARRAY_3(short)
static void _XMP_pack_array_3_UNSIGNED_SHORT	_XMP_SM_PACK_ARRAY_3(unsigned short)
static void _XMP_pack_array_3_INT		_XMP_SM_PACK_ARRAY_3(int)
static void _XMP_pack_array_3_UNSIGNED_INT	_XMP_SM_PACK_ARRAY_3(unsigned int)
static void _XMP_pack_array_3_LONG		_XMP_SM_PACK_ARRAY_3(long)
static void _XMP_pack_array_3_UNSIGNED_LONG	_XMP_SM_PACK_ARRAY_3(unsigned long)
static void _XMP_pack_array_3_LONGLONG		_XMP_SM_PACK_ARRAY_3(long long)
static void _XMP_pack_array_3_UNSIGNED_LONGLONG	_XMP_SM_PACK_ARRAY_3(unsigned long long)
static void _XMP_pack_array_3_FLOAT		_XMP_SM_PACK_ARRAY_3(float)
static void _XMP_pack_array_3_DOUBLE		_XMP_SM_PACK_ARRAY_3(double)
static void _XMP_pack_array_3_LONG_DOUBLE	_XMP_SM_PACK_ARRAY_3(long double)

static void _XMP_pack_array_3_GENERAL(void *buf_addr, void *src_addr, size_t type_size,
				      int *l, int *u, int *s, unsigned long long *d) {
  int src_lower0 = l[0]; int src_upper0 = u[0]; int src_stride0 = s[0];
  int src_lower1 = l[1]; int src_upper1 = u[1]; int src_stride1 = s[1]; unsigned long long src_dim_acc1 = d[1];
  int src_lower2 = l[2]; int src_upper2 = u[2]; int src_stride2 = s[2]; unsigned long long src_dim_acc2 = d[2];
  for (int k = src_lower2; k <= src_upper2; k += src_stride2) {
  for (int j = src_lower1; j <= src_upper1; j += src_stride1) {
    void *addr = (char *)src_addr + (type_size * ((k * src_dim_acc2) + (j * src_dim_acc1) + src_lower0));
    if (src_stride2 == 1) {
      size_t n = type_size * _XMP_M_COUNT_TRIPLETi(src_lower0, src_upper0, src_stride0);
      memcpy(buf_addr, addr, n);
      buf_addr = (char *)buf_addr + n;
    }
    else {
      int byte_stride = type_size * src_stride0;
      for (int i = src_lower0; i <= src_upper0; i += src_stride0) {
	memcpy(buf_addr, addr, type_size);
	buf_addr = (char *)buf_addr + type_size;
	addr = (char *)addr + byte_stride;
      }
    }
  }
  }
}

// --- dimension 4
#define _XMP_SM_PACK_ARRAY_4(_type) \
(_type *buf_addr, _type *src_addr, \
 int *l, int *u, int *s, unsigned long long *d) { \
  int src_lower0 = l[0]; int src_upper0 = u[0]; int src_stride0 = s[0]; \
  int src_lower1 = l[1]; int src_upper1 = u[1]; int src_stride1 = s[1]; unsigned long long src_dim_acc1 = d[1]; \
  int src_lower2 = l[2]; int src_upper2 = u[2]; int src_stride2 = s[2]; unsigned long long src_dim_acc2 = d[2]; \
  int src_lower3 = l[3]; int src_upper3 = u[3]; int src_stride3 = s[3]; unsigned long long src_dim_acc3 = d[3]; \
  for (int l = src_lower3; l <= src_upper3; l += src_stride3) { \
  for (int k = src_lower2; k <= src_upper2; k += src_stride2) { \
  for (int j = src_lower1; j <= src_upper1; j += src_stride1) { \
    _type *addr = src_addr + (l * src_dim_acc3) + (k * src_dim_acc2) + (j * src_dim_acc1); \
    for (int i = src_lower0; i <= src_upper0; i += src_stride0) {	\
      *buf_addr = addr[i];						\
      buf_addr++;							\
    }									\
  }									\
  }									\
  }									\
}

static void _XMP_pack_array_4_BOOL		_XMP_SM_PACK_ARRAY_4(_Bool)
static void _XMP_pack_array_4_CHAR		_XMP_SM_PACK_ARRAY_4(char)
static void _XMP_pack_array_4_UNSIGNED_CHAR	_XMP_SM_PACK_ARRAY_4(unsigned char)
static void _XMP_pack_array_4_SHORT		_XMP_SM_PACK_ARRAY_4(short)
static void _XMP_pack_array_4_UNSIGNED_SHORT	_XMP_SM_PACK_ARRAY_4(unsigned short)
static void _XMP_pack_array_4_INT		_XMP_SM_PACK_ARRAY_4(int)
static void _XMP_pack_array_4_UNSIGNED_INT	_XMP_SM_PACK_ARRAY_4(unsigned int)
static void _XMP_pack_array_4_LONG		_XMP_SM_PACK_ARRAY_4(long)
static void _XMP_pack_array_4_UNSIGNED_LONG	_XMP_SM_PACK_ARRAY_4(unsigned long)
static void _XMP_pack_array_4_LONGLONG		_XMP_SM_PACK_ARRAY_4(long long)
static void _XMP_pack_array_4_UNSIGNED_LONGLONG	_XMP_SM_PACK_ARRAY_4(unsigned long long)
static void _XMP_pack_array_4_FLOAT		_XMP_SM_PACK_ARRAY_4(float)
static void _XMP_pack_array_4_DOUBLE		_XMP_SM_PACK_ARRAY_4(double)
static void _XMP_pack_array_4_LONG_DOUBLE	_XMP_SM_PACK_ARRAY_4(long double)

static void _XMP_pack_array_4_GENERAL(void *buf_addr, void *src_addr, size_t type_size,
				      int *l, int *u, int *s, unsigned long long *d) {
  int src_lower0 = l[0]; int src_upper0 = u[0]; int src_stride0 = s[0];
  int src_lower1 = l[1]; int src_upper1 = u[1]; int src_stride1 = s[1]; unsigned long long src_dim_acc1 = d[1];
  int src_lower2 = l[2]; int src_upper2 = u[2]; int src_stride2 = s[2]; unsigned long long src_dim_acc2 = d[2];
  int src_lower3 = l[3]; int src_upper3 = u[3]; int src_stride3 = s[3]; unsigned long long src_dim_acc3 = d[3];
  for (int l = src_lower3; l <= src_upper3; l += src_stride3) {
  for (int k = src_lower2; k <= src_upper2; k += src_stride2) {
  for (int j = src_lower1; j <= src_upper1; j += src_stride1) {
    void *addr = (char *)src_addr + (type_size * ((l * src_dim_acc3) + (k * src_dim_acc2) + (j * src_dim_acc1) + src_lower0));
    if (src_stride0 == 1) {
      size_t n = type_size * _XMP_M_COUNT_TRIPLETi(src_lower0, src_upper0, src_stride0);
      memcpy(buf_addr, addr, n);
      buf_addr = (char *)buf_addr + n;
    }
    else {
      int byte_stride = type_size * src_stride0;
      for (int i = src_lower0; i <= src_upper0; i += src_stride0) {
	memcpy(buf_addr, addr, type_size);
	buf_addr = (char *)buf_addr + type_size;
	addr = (char *)addr + byte_stride;
      }
    }
  }
  }
  }
}

// --- dimension 5
#define _XMP_SM_PACK_ARRAY_5(_type) \
(_type *buf_addr, _type *src_addr, \
 int *l, int *u, int *s, unsigned long long *d) { \
  int src_lower0 = l[0]; int src_upper0 = u[0]; int src_stride0 = s[0]; \
  int src_lower1 = l[1]; int src_upper1 = u[1]; int src_stride1 = s[1]; unsigned long long src_dim_acc1 = d[1]; \
  int src_lower2 = l[2]; int src_upper2 = u[2]; int src_stride2 = s[2]; unsigned long long src_dim_acc2 = d[2]; \
  int src_lower3 = l[3]; int src_upper3 = u[3]; int src_stride3 = s[3]; unsigned long long src_dim_acc3 = d[3]; \
  int src_lower4 = l[4]; int src_upper4 = u[4]; int src_stride4 = s[4]; unsigned long long src_dim_acc4 = d[4]; \
  for (int m = src_lower4; m <= src_upper4; m += src_stride4) { \
  for (int l = src_lower3; l <= src_upper3; l += src_stride3) { \
  for (int k = src_lower2; k <= src_upper2; k += src_stride2) { \
  for (int j = src_lower1; j <= src_upper1; j += src_stride1) { \
    _type *addr = src_addr + (m * src_dim_acc4) + (l * src_dim_acc3) + (k * src_dim_acc2) + (j * src_dim_acc1); \
    for (int i = src_lower0; i <= src_upper0; i += src_stride0) {	\
      *buf_addr = addr[i];						\
      buf_addr++;							\
    }									\
  }									\
  }									\
  }									\
  }									\
}

static void _XMP_pack_array_5_BOOL		_XMP_SM_PACK_ARRAY_5(_Bool)
static void _XMP_pack_array_5_CHAR		_XMP_SM_PACK_ARRAY_5(char)
static void _XMP_pack_array_5_UNSIGNED_CHAR	_XMP_SM_PACK_ARRAY_5(unsigned char)
static void _XMP_pack_array_5_SHORT		_XMP_SM_PACK_ARRAY_5(short)
static void _XMP_pack_array_5_UNSIGNED_SHORT	_XMP_SM_PACK_ARRAY_5(unsigned short)
static void _XMP_pack_array_5_INT		_XMP_SM_PACK_ARRAY_5(int)
static void _XMP_pack_array_5_UNSIGNED_INT	_XMP_SM_PACK_ARRAY_5(unsigned int)
static void _XMP_pack_array_5_LONG		_XMP_SM_PACK_ARRAY_5(long)
static void _XMP_pack_array_5_UNSIGNED_LONG	_XMP_SM_PACK_ARRAY_5(unsigned long)
static void _XMP_pack_array_5_LONGLONG		_XMP_SM_PACK_ARRAY_5(long long)
static void _XMP_pack_array_5_UNSIGNED_LONGLONG	_XMP_SM_PACK_ARRAY_5(unsigned long long)
static void _XMP_pack_array_5_FLOAT		_XMP_SM_PACK_ARRAY_5(float)
static void _XMP_pack_array_5_DOUBLE		_XMP_SM_PACK_ARRAY_5(double)
static void _XMP_pack_array_5_LONG_DOUBLE	_XMP_SM_PACK_ARRAY_5(long double)

static void _XMP_pack_array_5_GENERAL(void *buf_addr, void *src_addr, size_t type_size,
				      int *l, int *u, int *s, unsigned long long *d) {
  int src_lower0 = l[0]; int src_upper0 = u[0]; int src_stride0 = s[0];
  int src_lower1 = l[1]; int src_upper1 = u[1]; int src_stride1 = s[1]; unsigned long long src_dim_acc1 = d[1];
  int src_lower2 = l[2]; int src_upper2 = u[2]; int src_stride2 = s[2]; unsigned long long src_dim_acc2 = d[2];
  int src_lower3 = l[3]; int src_upper3 = u[3]; int src_stride3 = s[3]; unsigned long long src_dim_acc3 = d[3];
  int src_lower4 = l[4]; int src_upper4 = u[4]; int src_stride4 = s[4]; unsigned long long src_dim_acc4 = d[4];
  for (int m = src_lower4; m <= src_upper4; m += src_stride4) {
  for (int l = src_lower3; l <= src_upper3; l += src_stride3) {
  for (int k = src_lower2; k <= src_upper2; k += src_stride2) {
  for (int j = src_lower1; j <= src_upper1; j += src_stride1) {
    void *addr = (char *)src_addr + (type_size * ((m * src_dim_acc4) + (l * src_dim_acc3) + (k * src_dim_acc2) +
						  (j * src_dim_acc1) + src_lower0));
    if (src_stride0 == 1) {
      size_t n = type_size * _XMP_M_COUNT_TRIPLETi(src_lower0, src_upper0, src_stride0);
      memcpy(buf_addr, addr, n);
      buf_addr = (char *)buf_addr + n;
    }
    else {
      int byte_stride = type_size * src_stride0;
      for (int i = src_lower0; i <= src_upper0; i += src_stride0) {
	memcpy(buf_addr, addr, type_size);
	buf_addr = (char *)buf_addr + type_size;
	addr = (char *)addr + byte_stride;
      }
    }
  }
  }
  }
  }
}

// --- dimension 6
#define _XMP_SM_PACK_ARRAY_6(_type) \
(_type *buf_addr, _type *src_addr, \
 int *l, int *u, int *s, unsigned long long *d) { \
  int src_lower0 = l[0]; int src_upper0 = u[0]; int src_stride0 = s[0]; \
  int src_lower1 = l[1]; int src_upper1 = u[1]; int src_stride1 = s[1]; unsigned long long src_dim_acc1 = d[1]; \
  int src_lower2 = l[2]; int src_upper2 = u[2]; int src_stride2 = s[2]; unsigned long long src_dim_acc2 = d[2]; \
  int src_lower3 = l[3]; int src_upper3 = u[3]; int src_stride3 = s[3]; unsigned long long src_dim_acc3 = d[3]; \
  int src_lower4 = l[4]; int src_upper4 = u[4]; int src_stride4 = s[4]; unsigned long long src_dim_acc4 = d[4]; \
  int src_lower5 = l[5]; int src_upper5 = u[5]; int src_stride5 = s[5]; unsigned long long src_dim_acc5 = d[5]; \
  for (int n = src_lower5; n <= src_upper5; n += src_stride5) { \
  for (int m = src_lower4; m <= src_upper4; m += src_stride4) { \
  for (int l = src_lower3; l <= src_upper3; l += src_stride3) { \
  for (int k = src_lower2; k <= src_upper2; k += src_stride2) { \
  for (int j = src_lower1; j <= src_upper1; j += src_stride1) { \
    _type *addr = src_addr + (n * src_dim_acc5) + (m * src_dim_acc4) + (l * src_dim_acc3) + \
                             (k * src_dim_acc2) + (j * src_dim_acc1);				\
    for (int i = src_lower0; i <= src_upper0; i += src_stride0) {	\
      *buf_addr = addr[i];						\
      buf_addr++;							\
    }									\
  }									\
  }									\
  }									\
  }									\
  }									\
}

static void _XMP_pack_array_6_BOOL		_XMP_SM_PACK_ARRAY_6(_Bool)
static void _XMP_pack_array_6_CHAR		_XMP_SM_PACK_ARRAY_6(char)
static void _XMP_pack_array_6_UNSIGNED_CHAR	_XMP_SM_PACK_ARRAY_6(unsigned char)
static void _XMP_pack_array_6_SHORT		_XMP_SM_PACK_ARRAY_6(short)
static void _XMP_pack_array_6_UNSIGNED_SHORT	_XMP_SM_PACK_ARRAY_6(unsigned short)
static void _XMP_pack_array_6_INT		_XMP_SM_PACK_ARRAY_6(int)
static void _XMP_pack_array_6_UNSIGNED_INT	_XMP_SM_PACK_ARRAY_6(unsigned int)
static void _XMP_pack_array_6_LONG		_XMP_SM_PACK_ARRAY_6(long)
static void _XMP_pack_array_6_UNSIGNED_LONG	_XMP_SM_PACK_ARRAY_6(unsigned long)
static void _XMP_pack_array_6_LONGLONG		_XMP_SM_PACK_ARRAY_6(long long)
static void _XMP_pack_array_6_UNSIGNED_LONGLONG	_XMP_SM_PACK_ARRAY_6(unsigned long long)
static void _XMP_pack_array_6_FLOAT		_XMP_SM_PACK_ARRAY_6(float)
static void _XMP_pack_array_6_DOUBLE		_XMP_SM_PACK_ARRAY_6(double)
static void _XMP_pack_array_6_LONG_DOUBLE	_XMP_SM_PACK_ARRAY_6(long double)

static void _XMP_pack_array_6_GENERAL(void *buf_addr, void *src_addr, size_t type_size,
				      int *l, int *u, int *s, unsigned long long *d) {
  int src_lower0 = l[0]; int src_upper0 = u[0]; int src_stride0 = s[0];
  int src_lower1 = l[1]; int src_upper1 = u[1]; int src_stride1 = s[1]; unsigned long long src_dim_acc1 = d[1];
  int src_lower2 = l[2]; int src_upper2 = u[2]; int src_stride2 = s[2]; unsigned long long src_dim_acc2 = d[2];
  int src_lower3 = l[3]; int src_upper3 = u[3]; int src_stride3 = s[3]; unsigned long long src_dim_acc3 = d[3];
  int src_lower4 = l[4]; int src_upper4 = u[4]; int src_stride4 = s[4]; unsigned long long src_dim_acc4 = d[4];
  int src_lower5 = l[5]; int src_upper5 = u[5]; int src_stride5 = s[5]; unsigned long long src_dim_acc5 = d[5];
  for (int n = src_lower5; n <= src_upper5; n += src_stride5) {
  for (int m = src_lower4; m <= src_upper4; m += src_stride4) {
  for (int l = src_lower3; l <= src_upper3; l += src_stride3) {
  for (int k = src_lower2; k <= src_upper2; k += src_stride2) {
  for (int j = src_lower1; j <= src_upper1; j += src_stride1) {
    void *addr = (char *)src_addr + (type_size * ((n * src_dim_acc5) + (m * src_dim_acc4) + (l * src_dim_acc3) +
                                                  (k * src_dim_acc2) + (j * src_dim_acc1) + src_lower0));
    if (src_stride0 == 1) {
      size_t n = type_size * _XMP_M_COUNT_TRIPLETi(src_lower0, src_upper0, src_stride0);
      memcpy(buf_addr, addr, n);
      buf_addr = (char *)buf_addr + n;
    }
    else {
      int byte_stride = type_size * src_stride0;
      for (int i = src_lower0; i <= src_upper0; i += src_stride0) {
	memcpy(buf_addr, addr, type_size);
	buf_addr = (char *)buf_addr + type_size;
	addr = (char *)addr + byte_stride;
      }
    }
  }
  }
  }
  }
  }
}

// --- dimension 7
#define _XMP_SM_PACK_ARRAY_7(_type) \
(_type *buf_addr, _type *src_addr, \
 int *l, int *u, int *s, unsigned long long *d) { \
  int src_lower0 = l[0]; int src_upper0 = u[0]; int src_stride0 = s[0]; \
  int src_lower1 = l[1]; int src_upper1 = u[1]; int src_stride1 = s[1]; unsigned long long src_dim_acc1 = d[1]; \
  int src_lower2 = l[2]; int src_upper2 = u[2]; int src_stride2 = s[2]; unsigned long long src_dim_acc2 = d[2]; \
  int src_lower3 = l[3]; int src_upper3 = u[3]; int src_stride3 = s[3]; unsigned long long src_dim_acc3 = d[3]; \
  int src_lower4 = l[4]; int src_upper4 = u[4]; int src_stride4 = s[4]; unsigned long long src_dim_acc4 = d[4]; \
  int src_lower5 = l[5]; int src_upper5 = u[5]; int src_stride5 = s[5]; unsigned long long src_dim_acc5 = d[5]; \
  int src_lower6 = l[6]; int src_upper6 = u[6]; int src_stride6 = s[6]; unsigned long long src_dim_acc6 = d[6]; \
  for (int o = src_lower6; o <= src_upper6; o += src_stride6) { \
  for (int n = src_lower5; n <= src_upper5; n += src_stride5) { \
  for (int m = src_lower4; m <= src_upper4; m += src_stride4) { \
  for (int l = src_lower3; l <= src_upper3; l += src_stride3) { \
  for (int k = src_lower2; k <= src_upper2; k += src_stride2) { \
  for (int j = src_lower1; j <= src_upper1; j += src_stride1) { \
    _type *addr = src_addr + (o * src_dim_acc6) + (n * src_dim_acc5) + (m * src_dim_acc4) + \
                             (l * src_dim_acc3) + (k * src_dim_acc2) + (j * src_dim_acc1); \
    for (int i = src_lower0; i <= src_upper0; i += src_stride0) {	\
      *buf_addr = addr[i];						\
      buf_addr++;							\
    }									\
  }									\
  }									\
  }									\
  }									\
  }									\
  }									\
}

static void _XMP_pack_array_7_BOOL		_XMP_SM_PACK_ARRAY_7(_Bool)
static void _XMP_pack_array_7_CHAR		_XMP_SM_PACK_ARRAY_7(char)
static void _XMP_pack_array_7_UNSIGNED_CHAR	_XMP_SM_PACK_ARRAY_7(unsigned char)
static void _XMP_pack_array_7_SHORT		_XMP_SM_PACK_ARRAY_7(short)
static void _XMP_pack_array_7_UNSIGNED_SHORT	_XMP_SM_PACK_ARRAY_7(unsigned short)
static void _XMP_pack_array_7_INT		_XMP_SM_PACK_ARRAY_7(int)
static void _XMP_pack_array_7_UNSIGNED_INT	_XMP_SM_PACK_ARRAY_7(unsigned int)
static void _XMP_pack_array_7_LONG		_XMP_SM_PACK_ARRAY_7(long)
static void _XMP_pack_array_7_UNSIGNED_LONG	_XMP_SM_PACK_ARRAY_7(unsigned long)
static void _XMP_pack_array_7_LONGLONG		_XMP_SM_PACK_ARRAY_7(long long)
static void _XMP_pack_array_7_UNSIGNED_LONGLONG	_XMP_SM_PACK_ARRAY_7(unsigned long long)
static void _XMP_pack_array_7_FLOAT		_XMP_SM_PACK_ARRAY_7(float)
static void _XMP_pack_array_7_DOUBLE		_XMP_SM_PACK_ARRAY_7(double)
static void _XMP_pack_array_7_LONG_DOUBLE	_XMP_SM_PACK_ARRAY_7(long double)

static void _XMP_pack_array_7_GENERAL(void *buf_addr, void *src_addr, size_t type_size,
				      int *l, int *u, int *s, unsigned long long *d) {
  int src_lower0 = l[0]; int src_upper0 = u[0]; int src_stride0 = s[0];
  int src_lower1 = l[1]; int src_upper1 = u[1]; int src_stride1 = s[1]; unsigned long long src_dim_acc1 = d[1];
  int src_lower2 = l[2]; int src_upper2 = u[2]; int src_stride2 = s[2]; unsigned long long src_dim_acc2 = d[2];
  int src_lower3 = l[3]; int src_upper3 = u[3]; int src_stride3 = s[3]; unsigned long long src_dim_acc3 = d[3];
  int src_lower4 = l[4]; int src_upper4 = u[4]; int src_stride4 = s[4]; unsigned long long src_dim_acc4 = d[4];
  int src_lower5 = l[5]; int src_upper5 = u[5]; int src_stride5 = s[5]; unsigned long long src_dim_acc5 = d[5];
  int src_lower6 = l[6]; int src_upper6 = u[6]; int src_stride6 = s[6]; unsigned long long src_dim_acc6 = d[6];
  for (int o = src_lower6; o <= src_upper6; o += src_stride6) {
  for (int n = src_lower5; n <= src_upper5; n += src_stride5) {
  for (int m = src_lower4; m <= src_upper4; m += src_stride4) {
  for (int l = src_lower3; l <= src_upper3; l += src_stride3) {
  for (int k = src_lower2; k <= src_upper2; k += src_stride2) {
  for (int j = src_lower1; j <= src_upper1; j += src_stride1) {
    void *addr = (char *)src_addr + (type_size * ((o * src_dim_acc6) + (n * src_dim_acc5) + (m * src_dim_acc4) +
						  (l * src_dim_acc3) + (k * src_dim_acc2) + (j * src_dim_acc1) + src_lower0));
    if (src_stride0 == 1) {
      size_t n = type_size * _XMP_M_COUNT_TRIPLETi(src_lower0, src_upper0, src_stride0);
      memcpy(buf_addr, addr, n);
      buf_addr = (char *)buf_addr + n;
    }
    else {
      int byte_stride = type_size * src_stride0;
      for (int i = src_lower0; i <= src_upper0; i += src_stride0) {
	memcpy(buf_addr, addr, type_size);
	buf_addr = (char *)buf_addr + type_size;
	addr = (char *)addr + byte_stride;
      }
    }
  }
  }
  }
  }
  }
  }
}


// ----- unpack array

// --- dimension 1
#define _XMP_SM_UNPACK_ARRAY_1(_type) \
(_type *dst_addr, _type *buf_addr, \
 int dst_lower, int dst_upper, int dst_stride) { \
  for (int i = dst_lower; i <= dst_upper; i += dst_stride) { \
    dst_addr[i] = *buf_addr; \
    buf_addr++; \
  } \
}

static void _XMP_unpack_array_1_BOOL			_XMP_SM_UNPACK_ARRAY_1(_Bool)
static void _XMP_unpack_array_1_CHAR			_XMP_SM_UNPACK_ARRAY_1(char)
static void _XMP_unpack_array_1_UNSIGNED_CHAR		_XMP_SM_UNPACK_ARRAY_1(unsigned char)
static void _XMP_unpack_array_1_SHORT			_XMP_SM_UNPACK_ARRAY_1(short)
static void _XMP_unpack_array_1_UNSIGNED_SHORT		_XMP_SM_UNPACK_ARRAY_1(unsigned short)
static void _XMP_unpack_array_1_INT			_XMP_SM_UNPACK_ARRAY_1(int)
static void _XMP_unpack_array_1_UNSIGNED_INT		_XMP_SM_UNPACK_ARRAY_1(unsigned int)
static void _XMP_unpack_array_1_LONG			_XMP_SM_UNPACK_ARRAY_1(long)
static void _XMP_unpack_array_1_UNSIGNED_LONG		_XMP_SM_UNPACK_ARRAY_1(unsigned long)
static void _XMP_unpack_array_1_LONGLONG		_XMP_SM_UNPACK_ARRAY_1(long long)
static void _XMP_unpack_array_1_UNSIGNED_LONGLONG	_XMP_SM_UNPACK_ARRAY_1(unsigned long long)
static void _XMP_unpack_array_1_FLOAT			_XMP_SM_UNPACK_ARRAY_1(float)
static void _XMP_unpack_array_1_DOUBLE			_XMP_SM_UNPACK_ARRAY_1(double)
static void _XMP_unpack_array_1_LONG_DOUBLE		_XMP_SM_UNPACK_ARRAY_1(long double)
static void _XMP_unpack_array_1_GENERAL(void *dst_addr, void *buf_addr, size_t type_size,
					int dst_lower, int dst_upper, int dst_stride) {
  if (dst_stride == 1) {
    memcpy((char *)dst_addr + (type_size * dst_lower), buf_addr,
           type_size * _XMP_M_COUNT_TRIPLETi(dst_lower, dst_upper, dst_stride));
  }
  else {
    void *addr = (char *)dst_addr + (type_size * dst_lower);
    int byte_stride = type_size * dst_stride;
    for (int i = dst_lower; i <= dst_upper; i += dst_stride) {
      memcpy(addr, buf_addr, type_size);
      addr = (char *)addr + byte_stride;
      buf_addr = (char *)buf_addr + type_size;
    }
  }
}

// --- dimension 2
#define _XMP_SM_UNPACK_ARRAY_2(_type) \
(_type *dst_addr, _type *buf_addr, \
 int *l, int *u, int *s, unsigned long long *d) { \
  int dst_lower0 = l[0]; int dst_upper0 = u[0]; int dst_stride0 = s[0]; \
  int dst_lower1 = l[1]; int dst_upper1 = u[1]; int dst_stride1 = s[1]; unsigned long long dst_dim_acc1 = d[1]; \
  for (int j = dst_lower1; j <= dst_upper1; j += dst_stride1) { \
    _type *addr = dst_addr + (j * dst_dim_acc1); \
    for (int i = dst_lower0; i <= dst_upper0; i += dst_stride0) { \
      addr[i] = *buf_addr; \
      buf_addr++; \
    } \
  } \
}

static void _XMP_unpack_array_2_BOOL			_XMP_SM_UNPACK_ARRAY_2(_Bool)
static void _XMP_unpack_array_2_CHAR			_XMP_SM_UNPACK_ARRAY_2(char)
static void _XMP_unpack_array_2_UNSIGNED_CHAR		_XMP_SM_UNPACK_ARRAY_2(unsigned char)
static void _XMP_unpack_array_2_SHORT			_XMP_SM_UNPACK_ARRAY_2(short)
static void _XMP_unpack_array_2_UNSIGNED_SHORT		_XMP_SM_UNPACK_ARRAY_2(unsigned short)
static void _XMP_unpack_array_2_INT			_XMP_SM_UNPACK_ARRAY_2(int)
static void _XMP_unpack_array_2_UNSIGNED_INT		_XMP_SM_UNPACK_ARRAY_2(unsigned int)
static void _XMP_unpack_array_2_LONG			_XMP_SM_UNPACK_ARRAY_2(long)
static void _XMP_unpack_array_2_UNSIGNED_LONG		_XMP_SM_UNPACK_ARRAY_2(unsigned long)
static void _XMP_unpack_array_2_LONGLONG		_XMP_SM_UNPACK_ARRAY_2(long long)
static void _XMP_unpack_array_2_UNSIGNED_LONGLONG	_XMP_SM_UNPACK_ARRAY_2(unsigned long long)
static void _XMP_unpack_array_2_FLOAT			_XMP_SM_UNPACK_ARRAY_2(float)
static void _XMP_unpack_array_2_DOUBLE			_XMP_SM_UNPACK_ARRAY_2(double)
static void _XMP_unpack_array_2_LONG_DOUBLE		_XMP_SM_UNPACK_ARRAY_2(long double)

static void _XMP_unpack_array_2_GENERAL(void *dst_addr, void *buf_addr, size_t type_size,
					int *l, int *u, int *s, unsigned long long *d) {
  int dst_lower0 = l[0]; int dst_upper0 = u[0]; int dst_stride0 = s[0];
  int dst_lower1 = l[1]; int dst_upper1 = u[1]; int dst_stride1 = s[1]; unsigned long long dst_dim_acc1 = d[1];
  for (int j = dst_lower1; j <= dst_upper1; j += dst_stride1) {
    void *addr = (char *)dst_addr + (type_size * ((j * dst_dim_acc1) + dst_lower0));
    if (dst_stride0 == 1) {
      size_t n = type_size * _XMP_M_COUNT_TRIPLETi(dst_lower0, dst_upper0, dst_stride0);
      memcpy(addr, buf_addr, n);
      buf_addr = (char *)buf_addr + n;
    }
    else {
      int byte_stride = type_size * dst_stride0;
      for (int i = dst_lower0; i <= dst_upper0; i += dst_stride0) {
        memcpy(addr, buf_addr, type_size);
        addr = (char *)addr + byte_stride;
        buf_addr = (char *)buf_addr + type_size;
      }
    }
  }
}

// --- dimension 3
#define _XMP_SM_UNPACK_ARRAY_3(_type) \
(_type *dst_addr, _type *buf_addr, \
 int *l, int *u, int *s, unsigned long long *d) { \
  int dst_lower0 = l[0]; int dst_upper0 = u[0]; int dst_stride0 = s[0]; \
  int dst_lower1 = l[1]; int dst_upper1 = u[1]; int dst_stride1 = s[1]; unsigned long long dst_dim_acc1 = d[1]; \
  int dst_lower2 = l[2]; int dst_upper2 = u[2]; int dst_stride2 = s[2]; unsigned long long dst_dim_acc2 = d[2]; \
  for (int k = dst_lower2; k <= dst_upper2; k += dst_stride2) { \
  for (int j = dst_lower1; j <= dst_upper1; j += dst_stride1) { \
    _type *addr = dst_addr + (k * dst_dim_acc2) + (j * dst_dim_acc1);	\
    for (int i = dst_lower0; i <= dst_upper0; i += dst_stride0) {	\
      addr[i] = *buf_addr;						\
      buf_addr++;							\
    }									\
  }									\
  }									\
}

static void _XMP_unpack_array_3_BOOL			_XMP_SM_UNPACK_ARRAY_3(_Bool)
static void _XMP_unpack_array_3_CHAR			_XMP_SM_UNPACK_ARRAY_3(char)
static void _XMP_unpack_array_3_UNSIGNED_CHAR		_XMP_SM_UNPACK_ARRAY_3(unsigned char)
static void _XMP_unpack_array_3_SHORT			_XMP_SM_UNPACK_ARRAY_3(short)
static void _XMP_unpack_array_3_UNSIGNED_SHORT		_XMP_SM_UNPACK_ARRAY_3(unsigned short)
static void _XMP_unpack_array_3_INT			_XMP_SM_UNPACK_ARRAY_3(int)
static void _XMP_unpack_array_3_UNSIGNED_INT		_XMP_SM_UNPACK_ARRAY_3(unsigned int)
static void _XMP_unpack_array_3_LONG			_XMP_SM_UNPACK_ARRAY_3(long)
static void _XMP_unpack_array_3_UNSIGNED_LONG		_XMP_SM_UNPACK_ARRAY_3(unsigned long)
static void _XMP_unpack_array_3_LONGLONG		_XMP_SM_UNPACK_ARRAY_3(long long)
static void _XMP_unpack_array_3_UNSIGNED_LONGLONG	_XMP_SM_UNPACK_ARRAY_3(unsigned long long)
static void _XMP_unpack_array_3_FLOAT			_XMP_SM_UNPACK_ARRAY_3(float)
static void _XMP_unpack_array_3_DOUBLE			_XMP_SM_UNPACK_ARRAY_3(double)
static void _XMP_unpack_array_3_LONG_DOUBLE		_XMP_SM_UNPACK_ARRAY_3(long double)

static void _XMP_unpack_array_3_GENERAL(void *dst_addr, void *buf_addr, size_t type_size,
					int *l, int *u, int *s, unsigned long long *d) {
  int dst_lower0 = l[0]; int dst_upper0 = u[0]; int dst_stride0 = s[0];
  int dst_lower1 = l[1]; int dst_upper1 = u[1]; int dst_stride1 = s[1]; unsigned long long dst_dim_acc1 = d[1];
  int dst_lower2 = l[2]; int dst_upper2 = u[2]; int dst_stride2 = s[2]; unsigned long long dst_dim_acc2 = d[2];
  for (int k = dst_lower2; k <= dst_upper2; k += dst_stride2) {
  for (int j = dst_lower1; j <= dst_upper1; j += dst_stride1) {
    void *addr = (char *)dst_addr + (type_size * ((k * dst_dim_acc2) + (j * dst_dim_acc1) + dst_lower0));
    if (dst_stride0 == 1) {
      size_t n = type_size * _XMP_M_COUNT_TRIPLETi(dst_lower0, dst_upper0, dst_stride0);
      memcpy(addr, buf_addr, n);
      buf_addr = (char *)buf_addr + n;
    }
    else {
      int byte_stride = type_size * dst_stride0;
      for (int i = dst_lower0; i <= dst_upper0; i += dst_stride0) {
	memcpy(addr, buf_addr, type_size);
	addr = (char *)addr + byte_stride;
	buf_addr = (char *)buf_addr + type_size;
      }
    }
  }
  }
}

// --- dimension 4
#define _XMP_SM_UNPACK_ARRAY_4(_type) \
(_type *dst_addr, _type *buf_addr, \
 int *l, int *u, int *s, unsigned long long *d) { \
  int dst_lower0 = l[0]; int dst_upper0 = u[0]; int dst_stride0 = s[0]; \
  int dst_lower1 = l[1]; int dst_upper1 = u[1]; int dst_stride1 = s[1]; unsigned long long dst_dim_acc1 = d[1]; \
  int dst_lower2 = l[2]; int dst_upper2 = u[2]; int dst_stride2 = s[2]; unsigned long long dst_dim_acc2 = d[2]; \
  int dst_lower3 = l[3]; int dst_upper3 = u[3]; int dst_stride3 = s[3]; unsigned long long dst_dim_acc3 = d[3]; \
  for (int l = dst_lower3; l <= dst_upper3; l += dst_stride3) { \
  for (int k = dst_lower2; k <= dst_upper2; k += dst_stride2) { \
  for (int j = dst_lower1; j <= dst_upper1; j += dst_stride1) { \
    _type *addr = dst_addr + (l * dst_dim_acc3) + (k * dst_dim_acc2) + (j * dst_dim_acc1); \
    for (int i = dst_lower0; i <= dst_upper0; i += dst_stride0) {	\
      addr[i] = *buf_addr;						\
      buf_addr++;							\
    }									\
  }									\
  }									\
  }									\
}

static void _XMP_unpack_array_4_BOOL			_XMP_SM_UNPACK_ARRAY_4(_Bool)
static void _XMP_unpack_array_4_CHAR			_XMP_SM_UNPACK_ARRAY_4(char)
static void _XMP_unpack_array_4_UNSIGNED_CHAR		_XMP_SM_UNPACK_ARRAY_4(unsigned char)
static void _XMP_unpack_array_4_SHORT			_XMP_SM_UNPACK_ARRAY_4(short)
static void _XMP_unpack_array_4_UNSIGNED_SHORT		_XMP_SM_UNPACK_ARRAY_4(unsigned short)
static void _XMP_unpack_array_4_INT			_XMP_SM_UNPACK_ARRAY_4(int)
static void _XMP_unpack_array_4_UNSIGNED_INT		_XMP_SM_UNPACK_ARRAY_4(unsigned int)
static void _XMP_unpack_array_4_LONG			_XMP_SM_UNPACK_ARRAY_4(long)
static void _XMP_unpack_array_4_UNSIGNED_LONG		_XMP_SM_UNPACK_ARRAY_4(unsigned long)
static void _XMP_unpack_array_4_LONGLONG			_XMP_SM_UNPACK_ARRAY_4(long long)
static void _XMP_unpack_array_4_UNSIGNED_LONGLONG	_XMP_SM_UNPACK_ARRAY_4(unsigned long long)
static void _XMP_unpack_array_4_FLOAT			_XMP_SM_UNPACK_ARRAY_4(float)
static void _XMP_unpack_array_4_DOUBLE			_XMP_SM_UNPACK_ARRAY_4(double)
static void _XMP_unpack_array_4_LONG_DOUBLE		_XMP_SM_UNPACK_ARRAY_4(long double)

static void _XMP_unpack_array_4_GENERAL(void *dst_addr, void *buf_addr, size_t type_size,
					int *l, int *u, int *s, unsigned long long *d) {
  int dst_lower0 = l[0]; int dst_upper0 = u[0]; int dst_stride0 = s[0];
  int dst_lower1 = l[1]; int dst_upper1 = u[1]; int dst_stride1 = s[1]; unsigned long long dst_dim_acc1 = d[1];
  int dst_lower2 = l[2]; int dst_upper2 = u[2]; int dst_stride2 = s[2]; unsigned long long dst_dim_acc2 = d[2];
  int dst_lower3 = l[3]; int dst_upper3 = u[3]; int dst_stride3 = s[3]; unsigned long long dst_dim_acc3 = d[3];
  for (int l = dst_lower3; l <= dst_upper3; l += dst_stride3) {
  for (int k = dst_lower2; k <= dst_upper2; k += dst_stride2) {
  for (int j = dst_lower1; j <= dst_upper1; j += dst_stride1) {
    void *addr = (char *)dst_addr + (type_size * ((l * dst_dim_acc3) + (k * dst_dim_acc2) + (j * dst_dim_acc1) + dst_lower0));
    if (dst_stride0 == 1) {
      size_t n = type_size * _XMP_M_COUNT_TRIPLETi(dst_lower0, dst_upper0, dst_stride0);
      memcpy(addr, buf_addr, n);
      buf_addr = (char *)buf_addr + n;
    }
    else {
      int byte_stride = type_size * dst_stride0;
      for (int i = dst_lower0; i <= dst_upper0; i += dst_stride0) {
	memcpy(addr, buf_addr, type_size);
	addr = (char *)addr + byte_stride;
	buf_addr = (char *)buf_addr + type_size;
      }
    }
  }
  }
  }
}

// --- dimension 5
#define _XMP_SM_UNPACK_ARRAY_5(_type) \
(_type *dst_addr, _type *buf_addr, \
 int *l, int *u, int *s, unsigned long long *d) { \
  int dst_lower0 = l[0]; int dst_upper0 = u[0]; int dst_stride0 = s[0]; \
  int dst_lower1 = l[1]; int dst_upper1 = u[1]; int dst_stride1 = s[1]; unsigned long long dst_dim_acc1 = d[1]; \
  int dst_lower2 = l[2]; int dst_upper2 = u[2]; int dst_stride2 = s[2]; unsigned long long dst_dim_acc2 = d[2]; \
  int dst_lower3 = l[3]; int dst_upper3 = u[3]; int dst_stride3 = s[3]; unsigned long long dst_dim_acc3 = d[3]; \
  int dst_lower4 = l[4]; int dst_upper4 = u[4]; int dst_stride4 = s[4]; unsigned long long dst_dim_acc4 = d[4]; \
  for (int m = dst_lower4; m <= dst_upper4; m += dst_stride4) { \
  for (int l = dst_lower3; l <= dst_upper3; l += dst_stride3) { \
  for (int k = dst_lower2; k <= dst_upper2; k += dst_stride2) { \
  for (int j = dst_lower1; j <= dst_upper1; j += dst_stride1) { \
    _type *addr = dst_addr + (m * dst_dim_acc4) + (l * dst_dim_acc3) + (k * dst_dim_acc2) + (j * dst_dim_acc1); \
    for (int i = dst_lower0; i <= dst_upper0; i += dst_stride0) {	\
      addr[i] = *buf_addr;						\
      buf_addr++;							\
    }									\
  }									\
  }									\
  }									\
  }									\
}

static void _XMP_unpack_array_5_BOOL			_XMP_SM_UNPACK_ARRAY_5(_Bool)
static void _XMP_unpack_array_5_CHAR			_XMP_SM_UNPACK_ARRAY_5(char)
static void _XMP_unpack_array_5_UNSIGNED_CHAR		_XMP_SM_UNPACK_ARRAY_5(unsigned char)
static void _XMP_unpack_array_5_SHORT			_XMP_SM_UNPACK_ARRAY_5(short)
static void _XMP_unpack_array_5_UNSIGNED_SHORT		_XMP_SM_UNPACK_ARRAY_5(unsigned short)
static void _XMP_unpack_array_5_INT			_XMP_SM_UNPACK_ARRAY_5(int)
static void _XMP_unpack_array_5_UNSIGNED_INT		_XMP_SM_UNPACK_ARRAY_5(unsigned int)
static void _XMP_unpack_array_5_LONG			_XMP_SM_UNPACK_ARRAY_5(long)
static void _XMP_unpack_array_5_UNSIGNED_LONG		_XMP_SM_UNPACK_ARRAY_5(unsigned long)
static void _XMP_unpack_array_5_LONGLONG			_XMP_SM_UNPACK_ARRAY_5(long long)
static void _XMP_unpack_array_5_UNSIGNED_LONGLONG	_XMP_SM_UNPACK_ARRAY_5(unsigned long long)
static void _XMP_unpack_array_5_FLOAT			_XMP_SM_UNPACK_ARRAY_5(float)
static void _XMP_unpack_array_5_DOUBLE			_XMP_SM_UNPACK_ARRAY_5(double)
static void _XMP_unpack_array_5_LONG_DOUBLE		_XMP_SM_UNPACK_ARRAY_5(long double)

static void _XMP_unpack_array_5_GENERAL(void *dst_addr, void *buf_addr, size_t type_size,
                                               int *l, int *u, int *s, unsigned long long *d) {
  int dst_lower0 = l[0]; int dst_upper0 = u[0]; int dst_stride0 = s[0];
  int dst_lower1 = l[1]; int dst_upper1 = u[1]; int dst_stride1 = s[1]; unsigned long long dst_dim_acc1 = d[1];
  int dst_lower2 = l[2]; int dst_upper2 = u[2]; int dst_stride2 = s[2]; unsigned long long dst_dim_acc2 = d[2];
  int dst_lower3 = l[3]; int dst_upper3 = u[3]; int dst_stride3 = s[3]; unsigned long long dst_dim_acc3 = d[3];
  int dst_lower4 = l[4]; int dst_upper4 = u[4]; int dst_stride4 = s[4]; unsigned long long dst_dim_acc4 = d[4];
  for (int m = dst_lower4; m <= dst_upper4; m += dst_stride4) {
  for (int l = dst_lower3; l <= dst_upper3; l += dst_stride3) {
  for (int k = dst_lower2; k <= dst_upper2; k += dst_stride2) {
  for (int j = dst_lower1; j <= dst_upper1; j += dst_stride1) {
    void *addr = (char *)dst_addr + (type_size * ((m * dst_dim_acc4) + (l * dst_dim_acc3) + (k * dst_dim_acc2) +
						  (j * dst_dim_acc1) + dst_lower0));
    if (dst_stride0 == 1) {
      size_t n = type_size * _XMP_M_COUNT_TRIPLETi(dst_lower0, dst_upper0, dst_stride0);
      memcpy(addr, buf_addr, n);
      buf_addr = (char *)buf_addr + n;
    }
    else {
      int byte_stride = type_size * dst_stride0;
      for (int i = dst_lower0; i <= dst_upper0; i += dst_stride0) {
	memcpy(addr, buf_addr, type_size);
	addr = (char *)addr + byte_stride;
	buf_addr = (char *)buf_addr + type_size;
      }
    }
  }
  }
  }
  }
}

// --- dimension 6
#define _XMP_SM_UNPACK_ARRAY_6(_type) \
(_type *dst_addr, _type *buf_addr, \
 int *l, int *u, int *s, unsigned long long *d) { \
  int dst_lower0 = l[0]; int dst_upper0 = u[0]; int dst_stride0 = s[0]; \
  int dst_lower1 = l[1]; int dst_upper1 = u[1]; int dst_stride1 = s[1]; unsigned long long dst_dim_acc1 = d[1]; \
  int dst_lower2 = l[2]; int dst_upper2 = u[2]; int dst_stride2 = s[2]; unsigned long long dst_dim_acc2 = d[2]; \
  int dst_lower3 = l[3]; int dst_upper3 = u[3]; int dst_stride3 = s[3]; unsigned long long dst_dim_acc3 = d[3]; \
  int dst_lower4 = l[4]; int dst_upper4 = u[4]; int dst_stride4 = s[4]; unsigned long long dst_dim_acc4 = d[4]; \
  int dst_lower5 = l[5]; int dst_upper5 = u[5]; int dst_stride5 = s[5]; unsigned long long dst_dim_acc5 = d[5]; \
  for (int n = dst_lower5; n <= dst_upper5; n += dst_stride5) { \
  for (int m = dst_lower4; m <= dst_upper4; m += dst_stride4) { \
  for (int l = dst_lower3; l <= dst_upper3; l += dst_stride3) { \
  for (int k = dst_lower2; k <= dst_upper2; k += dst_stride2) { \
  for (int j = dst_lower1; j <= dst_upper1; j += dst_stride1) { \
    _type *addr = dst_addr + (n * dst_dim_acc5) + (m * dst_dim_acc4) + (l * dst_dim_acc3) + \
                             (k * dst_dim_acc2) + (j * dst_dim_acc1); \
    for (int i = dst_lower0; i <= dst_upper0; i += dst_stride0) {     \
      addr[i] = *buf_addr;					      \
      buf_addr++;						      \
    }								      \
  }								      \
  }								      \
  }								      \
  }								      \
  }								      \
}

static void _XMP_unpack_array_6_BOOL			_XMP_SM_UNPACK_ARRAY_6(_Bool)
static void _XMP_unpack_array_6_CHAR			_XMP_SM_UNPACK_ARRAY_6(char)
static void _XMP_unpack_array_6_UNSIGNED_CHAR		_XMP_SM_UNPACK_ARRAY_6(unsigned char)
static void _XMP_unpack_array_6_SHORT			_XMP_SM_UNPACK_ARRAY_6(short)
static void _XMP_unpack_array_6_UNSIGNED_SHORT		_XMP_SM_UNPACK_ARRAY_6(unsigned short)
static void _XMP_unpack_array_6_INT			_XMP_SM_UNPACK_ARRAY_6(int)
static void _XMP_unpack_array_6_UNSIGNED_INT		_XMP_SM_UNPACK_ARRAY_6(unsigned int)
static void _XMP_unpack_array_6_LONG			_XMP_SM_UNPACK_ARRAY_6(long)
static void _XMP_unpack_array_6_UNSIGNED_LONG		_XMP_SM_UNPACK_ARRAY_6(unsigned long)
static void _XMP_unpack_array_6_LONGLONG			_XMP_SM_UNPACK_ARRAY_6(long long)
static void _XMP_unpack_array_6_UNSIGNED_LONGLONG	_XMP_SM_UNPACK_ARRAY_6(unsigned long long)
static void _XMP_unpack_array_6_FLOAT			_XMP_SM_UNPACK_ARRAY_6(float)
static void _XMP_unpack_array_6_DOUBLE			_XMP_SM_UNPACK_ARRAY_6(double)
static void _XMP_unpack_array_6_LONG_DOUBLE		_XMP_SM_UNPACK_ARRAY_6(long double)

static void _XMP_unpack_array_6_GENERAL(void *dst_addr, void *buf_addr, size_t type_size,
					int *l, int *u, int *s, unsigned long long *d) {
  int dst_lower0 = l[0]; int dst_upper0 = u[0]; int dst_stride0 = s[0];
  int dst_lower1 = l[1]; int dst_upper1 = u[1]; int dst_stride1 = s[1]; unsigned long long dst_dim_acc1 = d[1];
  int dst_lower2 = l[2]; int dst_upper2 = u[2]; int dst_stride2 = s[2]; unsigned long long dst_dim_acc2 = d[2];
  int dst_lower3 = l[3]; int dst_upper3 = u[3]; int dst_stride3 = s[3]; unsigned long long dst_dim_acc3 = d[3];
  int dst_lower4 = l[4]; int dst_upper4 = u[4]; int dst_stride4 = s[4]; unsigned long long dst_dim_acc4 = d[4];
  int dst_lower5 = l[5]; int dst_upper5 = u[5]; int dst_stride5 = s[5]; unsigned long long dst_dim_acc5 = d[5];
  for (int n = dst_lower5; n <= dst_upper5; n += dst_stride5) {
  for (int m = dst_lower4; m <= dst_upper4; m += dst_stride4) {
  for (int l = dst_lower3; l <= dst_upper3; l += dst_stride3) {
  for (int k = dst_lower2; k <= dst_upper2; k += dst_stride2) {
  for (int j = dst_lower1; j <= dst_upper1; j += dst_stride1) {
    void *addr = (char *)dst_addr + (type_size * ((n * dst_dim_acc5) + (m * dst_dim_acc4) + (l * dst_dim_acc3) +
                                                  (k * dst_dim_acc2) + (j * dst_dim_acc1) + dst_lower0));
    if (dst_stride0 == 1) {
      size_t n = type_size * _XMP_M_COUNT_TRIPLETi(dst_lower0, dst_upper0, dst_stride0);
      memcpy(addr, buf_addr, n);
      buf_addr = (char *)buf_addr + n;
    }
    else {
      int byte_stride = type_size * dst_stride0;
      for (int i = dst_lower0; i <= dst_upper0; i += dst_stride0) {
	memcpy(addr, buf_addr, type_size);
	addr = (char *)addr + byte_stride;
	buf_addr = (char *)buf_addr + type_size;
      }
    }
  }
  }
  }
  }
  }
}

// --- dimension 7
#define _XMP_SM_UNPACK_ARRAY_7(_type) \
(_type *dst_addr, _type *buf_addr, \
 int *l, int *u, int *s, unsigned long long *d) { \
  int dst_lower0 = l[0]; int dst_upper0 = u[0]; int dst_stride0 = s[0]; \
  int dst_lower1 = l[1]; int dst_upper1 = u[1]; int dst_stride1 = s[1]; unsigned long long dst_dim_acc1 = d[1]; \
  int dst_lower2 = l[2]; int dst_upper2 = u[2]; int dst_stride2 = s[2]; unsigned long long dst_dim_acc2 = d[2]; \
  int dst_lower3 = l[3]; int dst_upper3 = u[3]; int dst_stride3 = s[3]; unsigned long long dst_dim_acc3 = d[3]; \
  int dst_lower4 = l[4]; int dst_upper4 = u[4]; int dst_stride4 = s[4]; unsigned long long dst_dim_acc4 = d[4]; \
  int dst_lower5 = l[5]; int dst_upper5 = u[5]; int dst_stride5 = s[5]; unsigned long long dst_dim_acc5 = d[5]; \
  int dst_lower6 = l[6]; int dst_upper6 = u[6]; int dst_stride6 = s[6]; unsigned long long dst_dim_acc6 = d[6]; \
  for (int o = dst_lower6; o <= dst_upper6; o += dst_stride6) { \
  for (int n = dst_lower5; n <= dst_upper5; n += dst_stride5) { \
  for (int m = dst_lower4; m <= dst_upper4; m += dst_stride4) { \
  for (int l = dst_lower3; l <= dst_upper3; l += dst_stride3) { \
  for (int k = dst_lower2; k <= dst_upper2; k += dst_stride2) { \
  for (int j = dst_lower1; j <= dst_upper1; j += dst_stride1) { \
    _type *addr = dst_addr + (o * dst_dim_acc6) + (n * dst_dim_acc5) + (m * dst_dim_acc4) + \
                             (l * dst_dim_acc3) + (k * dst_dim_acc2) + (j * dst_dim_acc1);	\
    for (int i = dst_lower0; i <= dst_upper0; i += dst_stride0) {	\
      addr[i] = *buf_addr;						\
      buf_addr++;							\
    }									\
  }									\
  }									\
  }									\
  }									\
  }									\
  }									\
}

static void _XMP_unpack_array_7_BOOL			_XMP_SM_UNPACK_ARRAY_7(_Bool)
static void _XMP_unpack_array_7_CHAR			_XMP_SM_UNPACK_ARRAY_7(char)
static void _XMP_unpack_array_7_UNSIGNED_CHAR		_XMP_SM_UNPACK_ARRAY_7(unsigned char)
static void _XMP_unpack_array_7_SHORT			_XMP_SM_UNPACK_ARRAY_7(short)
static void _XMP_unpack_array_7_UNSIGNED_SHORT		_XMP_SM_UNPACK_ARRAY_7(unsigned short)
static void _XMP_unpack_array_7_INT			_XMP_SM_UNPACK_ARRAY_7(int)
static void _XMP_unpack_array_7_UNSIGNED_INT		_XMP_SM_UNPACK_ARRAY_7(unsigned int)
static void _XMP_unpack_array_7_LONG			_XMP_SM_UNPACK_ARRAY_7(long)
static void _XMP_unpack_array_7_UNSIGNED_LONG		_XMP_SM_UNPACK_ARRAY_7(unsigned long)
static void _XMP_unpack_array_7_LONGLONG			_XMP_SM_UNPACK_ARRAY_7(long long)
static void _XMP_unpack_array_7_UNSIGNED_LONGLONG	_XMP_SM_UNPACK_ARRAY_7(unsigned long long)
static void _XMP_unpack_array_7_FLOAT			_XMP_SM_UNPACK_ARRAY_7(float)
static void _XMP_unpack_array_7_DOUBLE			_XMP_SM_UNPACK_ARRAY_7(double)
static void _XMP_unpack_array_7_LONG_DOUBLE		_XMP_SM_UNPACK_ARRAY_7(long double)

static void _XMP_unpack_array_7_GENERAL(void *dst_addr, void *buf_addr, size_t type_size,
					int *l, int *u, int *s, unsigned long long *d) {
  int dst_lower0 = l[0]; int dst_upper0 = u[0]; int dst_stride0 = s[0];
  int dst_lower1 = l[1]; int dst_upper1 = u[1]; int dst_stride1 = s[1]; unsigned long long dst_dim_acc1 = d[1];
  int dst_lower2 = l[2]; int dst_upper2 = u[2]; int dst_stride2 = s[2]; unsigned long long dst_dim_acc2 = d[2];
  int dst_lower3 = l[3]; int dst_upper3 = u[3]; int dst_stride3 = s[3]; unsigned long long dst_dim_acc3 = d[3];
  int dst_lower4 = l[4]; int dst_upper4 = u[4]; int dst_stride4 = s[4]; unsigned long long dst_dim_acc4 = d[4];
  int dst_lower5 = l[5]; int dst_upper5 = u[5]; int dst_stride5 = s[5]; unsigned long long dst_dim_acc5 = d[5];
  int dst_lower6 = l[6]; int dst_upper6 = u[6]; int dst_stride6 = s[6]; unsigned long long dst_dim_acc6 = d[6];
  for (int o = dst_lower6; o <= dst_upper6; o += dst_stride6) {
  for (int n = dst_lower5; n <= dst_upper5; n += dst_stride5) {
  for (int m = dst_lower4; m <= dst_upper4; m += dst_stride4) {
  for (int l = dst_lower3; l <= dst_upper3; l += dst_stride3) {
  for (int k = dst_lower2; k <= dst_upper2; k += dst_stride2) {
  for (int j = dst_lower1; j <= dst_upper1; j += dst_stride1) {
    void *addr = (char *)dst_addr + (type_size * ((o * dst_dim_acc6) + (n * dst_dim_acc5) + (m * dst_dim_acc4) +
						  (l * dst_dim_acc3) + (k * dst_dim_acc2) + (j * dst_dim_acc1) + dst_lower0));
    if (dst_stride0 == 1) {
      size_t n = type_size * _XMP_M_COUNT_TRIPLETi(dst_lower0, dst_upper0, dst_stride0);
      memcpy(addr, buf_addr, n);
      buf_addr = (char *)buf_addr + n;
    }
    else {
      int byte_stride = type_size * dst_stride0;
      for (int i = dst_lower0; i <= dst_upper0; i += dst_stride0) {
	memcpy(addr, buf_addr, type_size);
	addr = (char *)addr + byte_stride;
	buf_addr = (char *)buf_addr + type_size;
      }
    }
  }
  }
  }
  }
  }
  }
}


void _XMP_normalize_array_section(int *lower, int *upper, int *stride) {
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
    l = *lower;
    u = *lower;
    s = 1;
   // _XMP_fatal("the stride of <array-section> is 0");
   // l = 0; u = 0; // XXX dummy
  }

  // normalize values
  if (s > 0) {
    u = u - ((u - l) % s);
    *upper = u;
    *stride = s;
  }
  else {
    s = -s;
    l = l + ((u - l) % s);
    *lower = l;
    *upper = u;
    *stride = s;
  }
}

void _XMP_pack_array_BASIC(void *buffer, void *src, int array_type,
                           int array_dim, int *l, int *u, int *s, unsigned long long *d) {
  switch (array_type) {
    case _XMP_N_TYPE_BOOL: {
        switch (array_dim) {
          case 1: _XMP_pack_array_1_BOOL(buffer, src, l[0], u[0], s[0]); break;
          case 2: _XMP_pack_array_2_BOOL(buffer, src, l, u, s, d); break;
          case 3: _XMP_pack_array_3_BOOL(buffer, src, l, u, s, d); break;
          case 4: _XMP_pack_array_4_BOOL(buffer, src, l, u, s, d); break;
          case 5: _XMP_pack_array_5_BOOL(buffer, src, l, u, s, d); break;
          case 6: _XMP_pack_array_6_BOOL(buffer, src, l, u, s, d); break;
          case 7: _XMP_pack_array_7_BOOL(buffer, src, l, u, s, d); break;
          default: _XMP_fatal("wrong array dimension");
        }
      } break;
    case _XMP_N_TYPE_CHAR: {
        switch (array_dim) {
          case 1: _XMP_pack_array_1_CHAR(buffer, src, l[0], u[0], s[0]); break;
          case 2: _XMP_pack_array_2_CHAR(buffer, src, l, u, s, d); break;
          case 3: _XMP_pack_array_3_CHAR(buffer, src, l, u, s, d); break;
          case 4: _XMP_pack_array_4_CHAR(buffer, src, l, u, s, d); break;
          case 5: _XMP_pack_array_5_CHAR(buffer, src, l, u, s, d); break;
          case 6: _XMP_pack_array_6_CHAR(buffer, src, l, u, s, d); break;
          case 7: _XMP_pack_array_7_CHAR(buffer, src, l, u, s, d); break;
          default: _XMP_fatal("wrong array dimension");
        }
      } break;
    case _XMP_N_TYPE_UNSIGNED_CHAR: {
        switch (array_dim) {
          case 1: _XMP_pack_array_1_UNSIGNED_CHAR(buffer, src, l[0], u[0], s[0]); break;
          case 2: _XMP_pack_array_2_UNSIGNED_CHAR(buffer, src, l, u, s, d); break;
          case 3: _XMP_pack_array_3_UNSIGNED_CHAR(buffer, src, l, u, s, d); break;
          case 4: _XMP_pack_array_4_UNSIGNED_CHAR(buffer, src, l, u, s, d); break;
          case 5: _XMP_pack_array_5_UNSIGNED_CHAR(buffer, src, l, u, s, d); break;
          case 6: _XMP_pack_array_6_UNSIGNED_CHAR(buffer, src, l, u, s, d); break;
          case 7: _XMP_pack_array_7_UNSIGNED_CHAR(buffer, src, l, u, s, d); break;
          default: _XMP_fatal("wrong array dimension");
        }
      } break;
    case _XMP_N_TYPE_SHORT: {
        switch (array_dim) {
          case 1: _XMP_pack_array_1_SHORT(buffer, src, l[0], u[0], s[0]); break;
          case 2: _XMP_pack_array_2_SHORT(buffer, src, l, u, s, d); break;
          case 3: _XMP_pack_array_3_SHORT(buffer, src, l, u, s, d); break;
          case 4: _XMP_pack_array_4_SHORT(buffer, src, l, u, s, d); break;
          case 5: _XMP_pack_array_5_SHORT(buffer, src, l, u, s, d); break;
          case 6: _XMP_pack_array_6_SHORT(buffer, src, l, u, s, d); break;
          case 7: _XMP_pack_array_7_SHORT(buffer, src, l, u, s, d); break;
          default: _XMP_fatal("wrong array dimension");
        }
      } break;
    case _XMP_N_TYPE_UNSIGNED_SHORT: {
        switch (array_dim) {
          case 1: _XMP_pack_array_1_UNSIGNED_SHORT(buffer, src, l[0], u[0], s[0]); break;
          case 2: _XMP_pack_array_2_UNSIGNED_SHORT(buffer, src, l, u, s, d); break;
          case 3: _XMP_pack_array_3_UNSIGNED_SHORT(buffer, src, l, u, s, d); break;
          case 4: _XMP_pack_array_4_UNSIGNED_SHORT(buffer, src, l, u, s, d); break;
          case 5: _XMP_pack_array_5_UNSIGNED_SHORT(buffer, src, l, u, s, d); break;
          case 6: _XMP_pack_array_6_UNSIGNED_SHORT(buffer, src, l, u, s, d); break;
          case 7: _XMP_pack_array_7_UNSIGNED_SHORT(buffer, src, l, u, s, d); break;
          default: _XMP_fatal("wrong array dimension");
        }
      } break;
    case _XMP_N_TYPE_INT: {
        switch (array_dim) {
          case 1: _XMP_pack_array_1_INT(buffer, src, l[0], u[0], s[0]); break;
          case 2: _XMP_pack_array_2_INT(buffer, src, l, u, s, d); break;
          case 3: _XMP_pack_array_3_INT(buffer, src, l, u, s, d); break;
          case 4: _XMP_pack_array_4_INT(buffer, src, l, u, s, d); break;
          case 5: _XMP_pack_array_5_INT(buffer, src, l, u, s, d); break;
          case 6: _XMP_pack_array_6_INT(buffer, src, l, u, s, d); break;
          case 7: _XMP_pack_array_7_INT(buffer, src, l, u, s, d); break;
          default: _XMP_fatal("wrong array dimension");
        }
      } break;
    case _XMP_N_TYPE_UNSIGNED_INT: {
        switch (array_dim) {
          case 1: _XMP_pack_array_1_UNSIGNED_INT(buffer, src, l[0], u[0], s[0]); break;
          case 2: _XMP_pack_array_2_UNSIGNED_INT(buffer, src, l, u, s, d); break;
          case 3: _XMP_pack_array_3_UNSIGNED_INT(buffer, src, l, u, s, d); break;
          case 4: _XMP_pack_array_4_UNSIGNED_INT(buffer, src, l, u, s, d); break;
          case 5: _XMP_pack_array_5_UNSIGNED_INT(buffer, src, l, u, s, d); break;
          case 6: _XMP_pack_array_6_UNSIGNED_INT(buffer, src, l, u, s, d); break;
          case 7: _XMP_pack_array_7_UNSIGNED_INT(buffer, src, l, u, s, d); break;
          default: _XMP_fatal("wrong array dimension");
        }
      } break;
    case _XMP_N_TYPE_LONG: {
        switch (array_dim) {
          case 1: _XMP_pack_array_1_LONG(buffer, src, l[0], u[0], s[0]); break;
          case 2: _XMP_pack_array_2_LONG(buffer, src, l, u, s, d); break;
          case 3: _XMP_pack_array_3_LONG(buffer, src, l, u, s, d); break;
          case 4: _XMP_pack_array_4_LONG(buffer, src, l, u, s, d); break;
          case 5: _XMP_pack_array_5_LONG(buffer, src, l, u, s, d); break;
          case 6: _XMP_pack_array_6_LONG(buffer, src, l, u, s, d); break;
          case 7: _XMP_pack_array_7_LONG(buffer, src, l, u, s, d); break;
          default: _XMP_fatal("wrong array dimension");
        }
      } break;
    case _XMP_N_TYPE_UNSIGNED_LONG: {
        switch (array_dim) {
          case 1: _XMP_pack_array_1_UNSIGNED_LONG(buffer, src, l[0], u[0], s[0]); break;
          case 2: _XMP_pack_array_2_UNSIGNED_LONG(buffer, src, l, u, s, d); break;
          case 3: _XMP_pack_array_3_UNSIGNED_LONG(buffer, src, l, u, s, d); break;
          case 4: _XMP_pack_array_4_UNSIGNED_LONG(buffer, src, l, u, s, d); break;
          case 5: _XMP_pack_array_5_UNSIGNED_LONG(buffer, src, l, u, s, d); break;
          case 6: _XMP_pack_array_6_UNSIGNED_LONG(buffer, src, l, u, s, d); break;
          case 7: _XMP_pack_array_7_UNSIGNED_LONG(buffer, src, l, u, s, d); break;
          default: _XMP_fatal("wrong array dimension");
        }
      } break;
    case _XMP_N_TYPE_LONGLONG: {
        switch (array_dim) {
          case 1: _XMP_pack_array_1_LONGLONG(buffer, src, l[0], u[0], s[0]); break;
          case 2: _XMP_pack_array_2_LONGLONG(buffer, src, l, u, s, d); break;
          case 3: _XMP_pack_array_3_LONGLONG(buffer, src, l, u, s, d); break;
          case 4: _XMP_pack_array_4_LONGLONG(buffer, src, l, u, s, d); break;
          case 5: _XMP_pack_array_5_LONGLONG(buffer, src, l, u, s, d); break;
          case 6: _XMP_pack_array_6_LONGLONG(buffer, src, l, u, s, d); break;
          case 7: _XMP_pack_array_7_LONGLONG(buffer, src, l, u, s, d); break;
          default: _XMP_fatal("wrong array dimension");
        }
      } break;
    case _XMP_N_TYPE_UNSIGNED_LONGLONG: {
        switch (array_dim) {
          case 1: _XMP_pack_array_1_UNSIGNED_LONGLONG(buffer, src, l[0], u[0], s[0]); break;
          case 2: _XMP_pack_array_2_UNSIGNED_LONGLONG(buffer, src, l, u, s, d); break;
          case 3: _XMP_pack_array_3_UNSIGNED_LONGLONG(buffer, src, l, u, s, d); break;
          case 4: _XMP_pack_array_4_UNSIGNED_LONGLONG(buffer, src, l, u, s, d); break;
          case 5: _XMP_pack_array_5_UNSIGNED_LONGLONG(buffer, src, l, u, s, d); break;
          case 6: _XMP_pack_array_6_UNSIGNED_LONGLONG(buffer, src, l, u, s, d); break;
          case 7: _XMP_pack_array_7_UNSIGNED_LONGLONG(buffer, src, l, u, s, d); break;
          default: _XMP_fatal("wrong array dimension");
        }
      } break;
    case _XMP_N_TYPE_FLOAT: {
        switch (array_dim) {
          case 1: _XMP_pack_array_1_FLOAT(buffer, src, l[0], u[0], s[0]); break;
          case 2: _XMP_pack_array_2_FLOAT(buffer, src, l, u, s, d); break;
          case 3: _XMP_pack_array_3_FLOAT(buffer, src, l, u, s, d); break;
          case 4: _XMP_pack_array_4_FLOAT(buffer, src, l, u, s, d); break;
          case 5: _XMP_pack_array_5_FLOAT(buffer, src, l, u, s, d); break;
          case 6: _XMP_pack_array_6_FLOAT(buffer, src, l, u, s, d); break;
          case 7: _XMP_pack_array_7_FLOAT(buffer, src, l, u, s, d); break;
          default: _XMP_fatal("wrong array dimension");
        }
      } break;
    case _XMP_N_TYPE_DOUBLE: {
        switch (array_dim) {
          case 1: _XMP_pack_array_1_DOUBLE(buffer, src, l[0], u[0], s[0]); break;
          case 2: _XMP_pack_array_2_DOUBLE(buffer, src, l, u, s, d); break;
          case 3: _XMP_pack_array_3_DOUBLE(buffer, src, l, u, s, d); break;
          case 4: _XMP_pack_array_4_DOUBLE(buffer, src, l, u, s, d); break;
          case 5: _XMP_pack_array_5_DOUBLE(buffer, src, l, u, s, d); break;
          case 6: _XMP_pack_array_6_DOUBLE(buffer, src, l, u, s, d); break;
          case 7: _XMP_pack_array_7_DOUBLE(buffer, src, l, u, s, d); break;
          default: _XMP_fatal("wrong array dimension");
        }
      } break;
    case _XMP_N_TYPE_LONG_DOUBLE: {
        switch (array_dim) {
          case 1: _XMP_pack_array_1_LONG_DOUBLE(buffer, src, l[0], u[0], s[0]); break;
          case 2: _XMP_pack_array_2_LONG_DOUBLE(buffer, src, l, u, s, d); break;
          case 3: _XMP_pack_array_3_LONG_DOUBLE(buffer, src, l, u, s, d); break;
          case 4: _XMP_pack_array_4_LONG_DOUBLE(buffer, src, l, u, s, d); break;
          case 5: _XMP_pack_array_5_LONG_DOUBLE(buffer, src, l, u, s, d); break;
          case 6: _XMP_pack_array_6_LONG_DOUBLE(buffer, src, l, u, s, d); break;
          case 7: _XMP_pack_array_7_LONG_DOUBLE(buffer, src, l, u, s, d); break;
          default: _XMP_fatal("wrong array dimension");
        }
      } break;
 // case _XMP_N_TYPE_FLOAT_IMAGINARY:
 //   _XMP_fatal("unknown data type for reflect");
 // case _XMP_N_TYPE_DOUBLE_IMAGINARY:
 //   _XMP_fatal("unknown data type for reflect");
 // case _XMP_N_TYPE_LONG_DOUBLE_IMAGINARY:
 //   _XMP_fatal("unknown data type for reflect");
 // case _XMP_N_TYPE_FLOAT_COMPLEX:
 //   _XMP_fatal("unknown data type for reflect");
 // case _XMP_N_TYPE_DOUBLE_COMPLEX:
 //   _XMP_fatal("unknown data type for reflect");
 // case _XMP_N_TYPE_LONG_DOUBLE_COMPLEX:
    default:
      _XMP_fatal("unknown data type for reflect");
  }
}

void _XMP_pack_array_GENERAL(void *buffer, void *src, size_t array_type_size,
                             int array_dim, int *l, int *u, int *s, unsigned long long *d) {
  switch (array_dim) {
    case 1: _XMP_pack_array_1_GENERAL(buffer, src, array_type_size, l[0], u[0], s[0]); break;
    case 2: _XMP_pack_array_2_GENERAL(buffer, src, array_type_size, l, u, s, d); break;
    case 3: _XMP_pack_array_3_GENERAL(buffer, src, array_type_size, l, u, s, d); break;
    case 4: _XMP_pack_array_4_GENERAL(buffer, src, array_type_size, l, u, s, d); break;
    case 5: _XMP_pack_array_5_GENERAL(buffer, src, array_type_size, l, u, s, d); break;
    case 6: _XMP_pack_array_6_GENERAL(buffer, src, array_type_size, l, u, s, d); break;
    case 7: _XMP_pack_array_7_GENERAL(buffer, src, array_type_size, l, u, s, d); break;
    default: _XMP_fatal("wrong array dimension");
  }
}

void _XMP_unpack_array_BASIC(void *dst, void *buffer, int array_type,
                             int array_dim, int *l, int *u, int *s, unsigned long long *d) {
  switch (array_type) {
    case _XMP_N_TYPE_BOOL: {
        switch (array_dim) {
          case 1: _XMP_unpack_array_1_BOOL(dst, buffer, l[0], u[0], s[0]); break;
          case 2: _XMP_unpack_array_2_BOOL(dst, buffer, l, u, s, d); break;
          case 3: _XMP_unpack_array_3_BOOL(dst, buffer, l, u, s, d); break;
          case 4: _XMP_unpack_array_4_BOOL(dst, buffer, l, u, s, d); break;
          case 5: _XMP_unpack_array_5_BOOL(dst, buffer, l, u, s, d); break;
          case 6: _XMP_unpack_array_6_BOOL(dst, buffer, l, u, s, d); break;
          case 7: _XMP_unpack_array_7_BOOL(dst, buffer, l, u, s, d); break;
          default: _XMP_fatal("wrong array dimension");
        }
      } break;
    case _XMP_N_TYPE_CHAR: {
        switch (array_dim) {
          case 1: _XMP_unpack_array_1_CHAR(dst, buffer, l[0], u[0], s[0]); break;
          case 2: _XMP_unpack_array_2_CHAR(dst, buffer, l, u, s, d); break;
          case 3: _XMP_unpack_array_3_CHAR(dst, buffer, l, u, s, d); break;
          case 4: _XMP_unpack_array_4_CHAR(dst, buffer, l, u, s, d); break;
          case 5: _XMP_unpack_array_5_CHAR(dst, buffer, l, u, s, d); break;
          case 6: _XMP_unpack_array_6_CHAR(dst, buffer, l, u, s, d); break;
          case 7: _XMP_unpack_array_7_CHAR(dst, buffer, l, u, s, d); break;
          default: _XMP_fatal("wrong array dimension");
        }
      } break;
    case _XMP_N_TYPE_UNSIGNED_CHAR: {
        switch (array_dim) {
          case 1: _XMP_unpack_array_1_UNSIGNED_CHAR(dst, buffer, l[0], u[0], s[0]); break;
          case 2: _XMP_unpack_array_2_UNSIGNED_CHAR(dst, buffer, l, u, s, d); break;
          case 3: _XMP_unpack_array_3_UNSIGNED_CHAR(dst, buffer, l, u, s, d); break;
          case 4: _XMP_unpack_array_4_UNSIGNED_CHAR(dst, buffer, l, u, s, d); break;
          case 5: _XMP_unpack_array_5_UNSIGNED_CHAR(dst, buffer, l, u, s, d); break;
          case 6: _XMP_unpack_array_6_UNSIGNED_CHAR(dst, buffer, l, u, s, d); break;
          case 7: _XMP_unpack_array_7_UNSIGNED_CHAR(dst, buffer, l, u, s, d); break;
          default: _XMP_fatal("wrong array dimension");
        }
      } break;
    case _XMP_N_TYPE_SHORT: {
        switch (array_dim) {
          case 1: _XMP_unpack_array_1_SHORT(dst, buffer, l[0], u[0], s[0]); break;
          case 2: _XMP_unpack_array_2_SHORT(dst, buffer, l, u, s, d); break;
          case 3: _XMP_unpack_array_3_SHORT(dst, buffer, l, u, s, d); break;
          case 4: _XMP_unpack_array_4_SHORT(dst, buffer, l, u, s, d); break;
          case 5: _XMP_unpack_array_5_SHORT(dst, buffer, l, u, s, d); break;
          case 6: _XMP_unpack_array_6_SHORT(dst, buffer, l, u, s, d); break;
          case 7: _XMP_unpack_array_7_SHORT(dst, buffer, l, u, s, d); break;
          default: _XMP_fatal("wrong array dimension");
        }
      } break;
    case _XMP_N_TYPE_UNSIGNED_SHORT: {
        switch (array_dim) {
          case 1: _XMP_unpack_array_1_UNSIGNED_SHORT(dst, buffer, l[0], u[0], s[0]); break;
          case 2: _XMP_unpack_array_2_UNSIGNED_SHORT(dst, buffer, l, u, s, d); break;
          case 3: _XMP_unpack_array_3_UNSIGNED_SHORT(dst, buffer, l, u, s, d); break;
          case 4: _XMP_unpack_array_4_UNSIGNED_SHORT(dst, buffer, l, u, s, d); break;
          case 5: _XMP_unpack_array_5_UNSIGNED_SHORT(dst, buffer, l, u, s, d); break;
          case 6: _XMP_unpack_array_6_UNSIGNED_SHORT(dst, buffer, l, u, s, d); break;
          case 7: _XMP_unpack_array_7_UNSIGNED_SHORT(dst, buffer, l, u, s, d); break;
          default: _XMP_fatal("wrong array dimension");
        }
      } break;
    case _XMP_N_TYPE_INT: {
        switch (array_dim) {
          case 1: _XMP_unpack_array_1_INT(dst, buffer, l[0], u[0], s[0]); break;
          case 2: _XMP_unpack_array_2_INT(dst, buffer, l, u, s, d); break;
          case 3: _XMP_unpack_array_3_INT(dst, buffer, l, u, s, d); break;
          case 4: _XMP_unpack_array_4_INT(dst, buffer, l, u, s, d); break;
          case 5: _XMP_unpack_array_5_INT(dst, buffer, l, u, s, d); break;
          case 6: _XMP_unpack_array_6_INT(dst, buffer, l, u, s, d); break;
          case 7: _XMP_unpack_array_7_INT(dst, buffer, l, u, s, d); break;
          default: _XMP_fatal("wrong array dimension");
        }
      } break;
    case _XMP_N_TYPE_UNSIGNED_INT: {
        switch (array_dim) {
          case 1: _XMP_unpack_array_1_UNSIGNED_INT(dst, buffer, l[0], u[0], s[0]); break;
          case 2: _XMP_unpack_array_2_UNSIGNED_INT(dst, buffer, l, u, s, d); break;
          case 3: _XMP_unpack_array_3_UNSIGNED_INT(dst, buffer, l, u, s, d); break;
          case 4: _XMP_unpack_array_4_UNSIGNED_INT(dst, buffer, l, u, s, d); break;
          case 5: _XMP_unpack_array_5_UNSIGNED_INT(dst, buffer, l, u, s, d); break;
          case 6: _XMP_unpack_array_6_UNSIGNED_INT(dst, buffer, l, u, s, d); break;
          case 7: _XMP_unpack_array_7_UNSIGNED_INT(dst, buffer, l, u, s, d); break;
          default: _XMP_fatal("wrong array dimension");
        }
      } break;
    case _XMP_N_TYPE_LONG: {
        switch (array_dim) {
          case 1: _XMP_unpack_array_1_LONG(dst, buffer, l[0], u[0], s[0]); break;
          case 2: _XMP_unpack_array_2_LONG(dst, buffer, l, u, s, d); break;
          case 3: _XMP_unpack_array_3_LONG(dst, buffer, l, u, s, d); break;
          case 4: _XMP_unpack_array_4_LONG(dst, buffer, l, u, s, d); break;
          case 5: _XMP_unpack_array_5_LONG(dst, buffer, l, u, s, d); break;
          case 6: _XMP_unpack_array_6_LONG(dst, buffer, l, u, s, d); break;
          case 7: _XMP_unpack_array_7_LONG(dst, buffer, l, u, s, d); break;
          default: _XMP_fatal("wrong array dimension");
        }
      } break;
    case _XMP_N_TYPE_UNSIGNED_LONG: {
        switch (array_dim) {
          case 1: _XMP_unpack_array_1_UNSIGNED_LONG(dst, buffer, l[0], u[0], s[0]); break;
          case 2: _XMP_unpack_array_2_UNSIGNED_LONG(dst, buffer, l, u, s, d); break;
          case 3: _XMP_unpack_array_3_UNSIGNED_LONG(dst, buffer, l, u, s, d); break;
          case 4: _XMP_unpack_array_4_UNSIGNED_LONG(dst, buffer, l, u, s, d); break;
          case 5: _XMP_unpack_array_5_UNSIGNED_LONG(dst, buffer, l, u, s, d); break;
          case 6: _XMP_unpack_array_6_UNSIGNED_LONG(dst, buffer, l, u, s, d); break;
          case 7: _XMP_unpack_array_7_UNSIGNED_LONG(dst, buffer, l, u, s, d); break;
          default: _XMP_fatal("wrong array dimension");
        }
      } break;
    case _XMP_N_TYPE_LONGLONG: {
        switch (array_dim) {
          case 1: _XMP_unpack_array_1_LONGLONG(dst, buffer, l[0], u[0], s[0]); break;
          case 2: _XMP_unpack_array_2_LONGLONG(dst, buffer, l, u, s, d); break;
          case 3: _XMP_unpack_array_3_LONGLONG(dst, buffer, l, u, s, d); break;
          case 4: _XMP_unpack_array_4_LONGLONG(dst, buffer, l, u, s, d); break;
          case 5: _XMP_unpack_array_5_LONGLONG(dst, buffer, l, u, s, d); break;
          case 6: _XMP_unpack_array_6_LONGLONG(dst, buffer, l, u, s, d); break;
          case 7: _XMP_unpack_array_7_LONGLONG(dst, buffer, l, u, s, d); break;
          default: _XMP_fatal("wrong array dimension");
        }
      } break;
    case _XMP_N_TYPE_UNSIGNED_LONGLONG: {
        switch (array_dim) {
          case 1: _XMP_unpack_array_1_UNSIGNED_LONGLONG(dst, buffer, l[0], u[0], s[0]); break;
          case 2: _XMP_unpack_array_2_UNSIGNED_LONGLONG(dst, buffer, l, u, s, d); break;
          case 3: _XMP_unpack_array_3_UNSIGNED_LONGLONG(dst, buffer, l, u, s, d); break;
          case 4: _XMP_unpack_array_4_UNSIGNED_LONGLONG(dst, buffer, l, u, s, d); break;
          case 5: _XMP_unpack_array_5_UNSIGNED_LONGLONG(dst, buffer, l, u, s, d); break;
          case 6: _XMP_unpack_array_6_UNSIGNED_LONGLONG(dst, buffer, l, u, s, d); break;
          case 7: _XMP_unpack_array_7_UNSIGNED_LONGLONG(dst, buffer, l, u, s, d); break;
          default: _XMP_fatal("wrong array dimension");
        }
      } break;
    case _XMP_N_TYPE_FLOAT: {
        switch (array_dim) {
          case 1: _XMP_unpack_array_1_FLOAT(dst, buffer, l[0], u[0], s[0]); break;
          case 2: _XMP_unpack_array_2_FLOAT(dst, buffer, l, u, s, d); break;
          case 3: _XMP_unpack_array_3_FLOAT(dst, buffer, l, u, s, d); break;
          case 4: _XMP_unpack_array_4_FLOAT(dst, buffer, l, u, s, d); break;
          case 5: _XMP_unpack_array_5_FLOAT(dst, buffer, l, u, s, d); break;
          case 6: _XMP_unpack_array_6_FLOAT(dst, buffer, l, u, s, d); break;
          case 7: _XMP_unpack_array_7_FLOAT(dst, buffer, l, u, s, d); break;
          default: _XMP_fatal("wrong array dimension");
        }
      } break;
    case _XMP_N_TYPE_DOUBLE: {
        switch (array_dim) {
          case 1: _XMP_unpack_array_1_DOUBLE(dst, buffer, l[0], u[0], s[0]); break;
          case 2: _XMP_unpack_array_2_DOUBLE(dst, buffer, l, u, s, d); break;
          case 3: _XMP_unpack_array_3_DOUBLE(dst, buffer, l, u, s, d); break;
          case 4: _XMP_unpack_array_4_DOUBLE(dst, buffer, l, u, s, d); break;
          case 5: _XMP_unpack_array_5_DOUBLE(dst, buffer, l, u, s, d); break;
          case 6: _XMP_unpack_array_6_DOUBLE(dst, buffer, l, u, s, d); break;
          case 7: _XMP_unpack_array_7_DOUBLE(dst, buffer, l, u, s, d); break;
          default: _XMP_fatal("wrong array dimension");
        }
      } break;
    case _XMP_N_TYPE_LONG_DOUBLE: {
        switch (array_dim) {
          case 1: _XMP_unpack_array_1_LONG_DOUBLE(dst, buffer, l[0], u[0], s[0]); break;
          case 2: _XMP_unpack_array_2_LONG_DOUBLE(dst, buffer, l, u, s, d); break;
          case 3: _XMP_unpack_array_3_LONG_DOUBLE(dst, buffer, l, u, s, d); break;
          case 4: _XMP_unpack_array_4_LONG_DOUBLE(dst, buffer, l, u, s, d); break;
          case 5: _XMP_unpack_array_5_LONG_DOUBLE(dst, buffer, l, u, s, d); break;
          case 6: _XMP_unpack_array_6_LONG_DOUBLE(dst, buffer, l, u, s, d); break;
          case 7: _XMP_unpack_array_7_LONG_DOUBLE(dst, buffer, l, u, s, d); break;
          default: _XMP_fatal("wrong array dimension");
        }
      } break;
 // case _XMP_N_TYPE_FLOAT_IMAGINARY:
 //   _XMP_fatal("unknown data type for reflect");
 // case _XMP_N_TYPE_DOUBLE_IMAGINARY:
 //   _XMP_fatal("unknown data type for reflect");
 // case _XMP_N_TYPE_LONG_DOUBLE_IMAGINARY:
 //   _XMP_fatal("unknown data type for reflect");
 // case _XMP_N_TYPE_FLOAT_COMPLEX:
 //   _XMP_fatal("unknown data type for reflect");
 // case _XMP_N_TYPE_DOUBLE_COMPLEX:
 //   _XMP_fatal("unknown data type for reflect");
 // case _XMP_N_TYPE_LONG_DOUBLE_COMPLEX:
 //   _XMP_fatal("unknown data type for reflect");
    default:
      _XMP_fatal("unknown data type for reflect");
  }
}

void _XMP_unpack_array_GENERAL(void *dst, void *buffer, size_t array_type_size,
                               int array_dim, int *l, int *u, int *s, unsigned long long *d) {
  switch (array_dim) {
    case 1: _XMP_unpack_array_1_GENERAL(dst, buffer, array_type_size, l[0], u[0], s[0]); break;
    case 2: _XMP_unpack_array_2_GENERAL(dst, buffer, array_type_size, l, u, s, d); break;
    case 3: _XMP_unpack_array_3_GENERAL(dst, buffer, array_type_size, l, u, s, d); break;
    case 4: _XMP_unpack_array_4_GENERAL(dst, buffer, array_type_size, l, u, s, d); break;
    case 5: _XMP_unpack_array_5_GENERAL(dst, buffer, array_type_size, l, u, s, d); break;
    case 6: _XMP_unpack_array_6_GENERAL(dst, buffer, array_type_size, l, u, s, d); break;
    case 7: _XMP_unpack_array_7_GENERAL(dst, buffer, array_type_size, l, u, s, d); break;
    default: _XMP_fatal("wrong array dimension");
  }
}

void _XMP_pack_array(void *buffer, void *src, int array_type, size_t array_type_size,
                     int array_dim, int *l, int *u, int *s, unsigned long long *d) {
  if (array_type == _XMP_N_TYPE_NONBASIC) {
    _XMP_pack_array_GENERAL(buffer, src, array_type_size, array_dim, l, u, s, d);
  } else {
    _XMP_pack_array_BASIC(buffer, src, array_type, array_dim, l, u, s, d);
  }
}

void _XMP_unpack_array(void *dst, void *buffer, int array_type, size_t array_type_size,
                       int array_dim, int *l, int *u, int *s, unsigned long long *d) {
  if (array_type == _XMP_N_TYPE_NONBASIC) {
    _XMP_unpack_array_GENERAL(dst, buffer, array_type_size, array_dim, l, u, s, d);
  } else {
    _XMP_unpack_array_BASIC(dst, buffer, array_type, array_dim, l, u, s, d);
  }
}

/*
void _XMP_pack_array_2_DOUBLE(double *buf_addr, double *src_addr,
			      int *l, int *u, int *s, unsigned long long *d)
{

  int src_lower0 = l[0]; int src_upper0 = u[0]; int src_stride0 = s[0];
  int src_lower1 = l[1]; int src_upper1 = u[1]; int src_stride1 = s[1];
  unsigned long long src_dim_acc1 = d[1];

  for (int j = src_lower1; j <= src_upper1; j += src_stride1) {
    double *addr = src_addr + (j * src_dim_acc1);

    for (int i = src_lower0; i <= src_upper0; i += src_stride0) {
      *buf_addr = addr[i];
      buf_addr++;
    }
  }

}


void _XMP_unpack_array_2_DOUBLE(double *dst_addr, double *buf_addr,
				int *l, int *u, int *s, unsigned long long *d)
{
  int dst_lower0 = l[0]; int dst_upper0 = u[0]; int dst_stride0 = s[0];
  int dst_lower1 = l[1]; int dst_upper1 = u[1]; int dst_stride1 = s[1];
  unsigned long long dst_dim_acc1 = d[1];

  for (int j = dst_lower1; j <= dst_upper1; j += dst_stride1) {
    double *addr = dst_addr + (j * dst_dim_acc1);

    for (int i = dst_lower0; i <= dst_upper0; i += dst_stride0) {
      addr[i] = *buf_addr;
      buf_addr++;
    }
  }

}
*/
