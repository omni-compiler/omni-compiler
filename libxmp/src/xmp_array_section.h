#ifndef _XCALABLEMP_ARRAY_SECTION
#define _XCALABLEMP_ARRAY_SECTION

// ----- pack array
// --- dimension 1
#define _XCALABLEMP_SM_PACK_ARRAY_1(_type) \
(_type *buf_addr, _type *src_addr, \
 int src_lower, int src_upper, int src_stride);

extern void _XCALABLEMP_pack_array_1_BOOL		_XCALABLEMP_SM_PACK_ARRAY_1(_Bool)
extern void _XCALABLEMP_pack_array_1_CHAR		_XCALABLEMP_SM_PACK_ARRAY_1(char)
extern void _XCALABLEMP_pack_array_1_UNSIGNED_CHAR	_XCALABLEMP_SM_PACK_ARRAY_1(unsigned char)
extern void _XCALABLEMP_pack_array_1_SHORT		_XCALABLEMP_SM_PACK_ARRAY_1(short)
extern void _XCALABLEMP_pack_array_1_UNSIGNED_SHORT	_XCALABLEMP_SM_PACK_ARRAY_1(unsigned short)
extern void _XCALABLEMP_pack_array_1_INT		_XCALABLEMP_SM_PACK_ARRAY_1(int)
extern void _XCALABLEMP_pack_array_1_UNSIGNED_INT	_XCALABLEMP_SM_PACK_ARRAY_1(unsigned int)
extern void _XCALABLEMP_pack_array_1_LONG		_XCALABLEMP_SM_PACK_ARRAY_1(long)
extern void _XCALABLEMP_pack_array_1_UNSIGNED_LONG	_XCALABLEMP_SM_PACK_ARRAY_1(unsigned long)
extern void _XCALABLEMP_pack_array_1_LONG_LONG		_XCALABLEMP_SM_PACK_ARRAY_1(long long)
extern void _XCALABLEMP_pack_array_1_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_PACK_ARRAY_1(unsigned long long)
extern void _XCALABLEMP_pack_array_1_FLOAT		_XCALABLEMP_SM_PACK_ARRAY_1(float)
extern void _XCALABLEMP_pack_array_1_DOUBLE		_XCALABLEMP_SM_PACK_ARRAY_1(double)
extern void _XCALABLEMP_pack_array_1_LONG_DOUBLE	_XCALABLEMP_SM_PACK_ARRAY_1(long double)

// --- dimension 2
#define _XCALABLEMP_SM_PACK_ARRAY_2(_type) \
(_type *buf_addr, _type *src_addr, \
 int src_lower0, int src_upper0, int src_stride0, unsigned long long src_dim_acc0, \
 int src_lower1, int src_upper1, int src_stride1);

extern void _XCALABLEMP_pack_array_2_BOOL		_XCALABLEMP_SM_PACK_ARRAY_2(_Bool)
extern void _XCALABLEMP_pack_array_2_CHAR		_XCALABLEMP_SM_PACK_ARRAY_2(char)
extern void _XCALABLEMP_pack_array_2_UNSIGNED_CHAR	_XCALABLEMP_SM_PACK_ARRAY_2(unsigned char)
extern void _XCALABLEMP_pack_array_2_SHORT		_XCALABLEMP_SM_PACK_ARRAY_2(short)
extern void _XCALABLEMP_pack_array_2_UNSIGNED_SHORT	_XCALABLEMP_SM_PACK_ARRAY_2(unsigned short)
extern void _XCALABLEMP_pack_array_2_INT		_XCALABLEMP_SM_PACK_ARRAY_2(int)
extern void _XCALABLEMP_pack_array_2_UNSIGNED_INT	_XCALABLEMP_SM_PACK_ARRAY_2(unsigned int)
extern void _XCALABLEMP_pack_array_2_LONG		_XCALABLEMP_SM_PACK_ARRAY_2(long)
extern void _XCALABLEMP_pack_array_2_UNSIGNED_LONG	_XCALABLEMP_SM_PACK_ARRAY_2(unsigned long)
extern void _XCALABLEMP_pack_array_2_LONG_LONG		_XCALABLEMP_SM_PACK_ARRAY_2(long long)
extern void _XCALABLEMP_pack_array_2_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_PACK_ARRAY_2(unsigned long long)
extern void _XCALABLEMP_pack_array_2_FLOAT		_XCALABLEMP_SM_PACK_ARRAY_2(float)
extern void _XCALABLEMP_pack_array_2_DOUBLE		_XCALABLEMP_SM_PACK_ARRAY_2(double)
extern void _XCALABLEMP_pack_array_2_LONG_DOUBLE	_XCALABLEMP_SM_PACK_ARRAY_2(long double)

// --- dimension 3
#define _XCALABLEMP_SM_PACK_ARRAY_3(_type) \
(_type *buf_addr, const _type *const src_addr, \
 int src_lower0, int src_upper0, int src_stride0, const unsigned long long src_dim_acc0, \
 int src_lower1, int src_upper1, int src_stride1, const unsigned long long src_dim_acc1, \
 int src_lower2, int src_upper2, int src_stride2);

extern void _XCALABLEMP_pack_array_3_BOOL		_XCALABLEMP_SM_PACK_ARRAY_3(_Bool)
extern void _XCALABLEMP_pack_array_3_CHAR		_XCALABLEMP_SM_PACK_ARRAY_3(char)
extern void _XCALABLEMP_pack_array_3_UNSIGNED_CHAR	_XCALABLEMP_SM_PACK_ARRAY_3(unsigned char)
extern void _XCALABLEMP_pack_array_3_SHORT		_XCALABLEMP_SM_PACK_ARRAY_3(short)
extern void _XCALABLEMP_pack_array_3_UNSIGNED_SHORT	_XCALABLEMP_SM_PACK_ARRAY_3(unsigned short)
extern void _XCALABLEMP_pack_array_3_INT		_XCALABLEMP_SM_PACK_ARRAY_3(int)
extern void _XCALABLEMP_pack_array_3_UNSIGNED_INT	_XCALABLEMP_SM_PACK_ARRAY_3(unsigned int)
extern void _XCALABLEMP_pack_array_3_LONG		_XCALABLEMP_SM_PACK_ARRAY_3(long)
extern void _XCALABLEMP_pack_array_3_UNSIGNED_LONG	_XCALABLEMP_SM_PACK_ARRAY_3(unsigned long)
extern void _XCALABLEMP_pack_array_3_LONG_LONG		_XCALABLEMP_SM_PACK_ARRAY_3(long long)
extern void _XCALABLEMP_pack_array_3_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_PACK_ARRAY_3(unsigned long long)
extern void _XCALABLEMP_pack_array_3_FLOAT		_XCALABLEMP_SM_PACK_ARRAY_3(float)
extern void _XCALABLEMP_pack_array_3_DOUBLE		_XCALABLEMP_SM_PACK_ARRAY_3(double)
extern void _XCALABLEMP_pack_array_3_LONG_DOUBLE	_XCALABLEMP_SM_PACK_ARRAY_3(long double)

// --- dimension 4
#define _XCALABLEMP_SM_PACK_ARRAY_4(_type) \
(_type *buf_addr, const _type *const src_addr, \
 int src_lower0, int src_upper0, int src_stride0, const unsigned long long src_dim_acc0, \
 int src_lower1, int src_upper1, int src_stride1, const unsigned long long src_dim_acc1, \
 int src_lower2, int src_upper2, int src_stride2, const unsigned long long src_dim_acc2, \
 int src_lower3, int src_upper3, int src_stride3);

extern void _XCALABLEMP_pack_array_4_BOOL		_XCALABLEMP_SM_PACK_ARRAY_4(_Bool)
extern void _XCALABLEMP_pack_array_4_CHAR		_XCALABLEMP_SM_PACK_ARRAY_4(char)
extern void _XCALABLEMP_pack_array_4_UNSIGNED_CHAR	_XCALABLEMP_SM_PACK_ARRAY_4(unsigned char)
extern void _XCALABLEMP_pack_array_4_SHORT		_XCALABLEMP_SM_PACK_ARRAY_4(short)
extern void _XCALABLEMP_pack_array_4_UNSIGNED_SHORT	_XCALABLEMP_SM_PACK_ARRAY_4(unsigned short)
extern void _XCALABLEMP_pack_array_4_INT		_XCALABLEMP_SM_PACK_ARRAY_4(int)
extern void _XCALABLEMP_pack_array_4_UNSIGNED_INT	_XCALABLEMP_SM_PACK_ARRAY_4(unsigned int)
extern void _XCALABLEMP_pack_array_4_LONG		_XCALABLEMP_SM_PACK_ARRAY_4(long)
extern void _XCALABLEMP_pack_array_4_UNSIGNED_LONG	_XCALABLEMP_SM_PACK_ARRAY_4(unsigned long)
extern void _XCALABLEMP_pack_array_4_LONG_LONG		_XCALABLEMP_SM_PACK_ARRAY_4(long long)
extern void _XCALABLEMP_pack_array_4_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_PACK_ARRAY_4(unsigned long long)
extern void _XCALABLEMP_pack_array_4_FLOAT		_XCALABLEMP_SM_PACK_ARRAY_4(float)
extern void _XCALABLEMP_pack_array_4_DOUBLE		_XCALABLEMP_SM_PACK_ARRAY_4(double)
extern void _XCALABLEMP_pack_array_4_LONG_DOUBLE	_XCALABLEMP_SM_PACK_ARRAY_4(long double)

// --- dimension 5
#define _XCALABLEMP_SM_PACK_ARRAY_5(_type) \
(_type *buf_addr, const _type *const src_addr, \
 int src_lower0, int src_upper0, int src_stride0, const unsigned long long src_dim_acc0, \
 int src_lower1, int src_upper1, int src_stride1, const unsigned long long src_dim_acc1, \
 int src_lower2, int src_upper2, int src_stride2, const unsigned long long src_dim_acc2, \
 int src_lower3, int src_upper3, int src_stride3, const unsigned long long src_dim_acc3, \
 int src_lower4, int src_upper4, int src_stride4);

extern void _XCALABLEMP_pack_array_5_BOOL		_XCALABLEMP_SM_PACK_ARRAY_5(_Bool)
extern void _XCALABLEMP_pack_array_5_CHAR		_XCALABLEMP_SM_PACK_ARRAY_5(char)
extern void _XCALABLEMP_pack_array_5_UNSIGNED_CHAR	_XCALABLEMP_SM_PACK_ARRAY_5(unsigned char)
extern void _XCALABLEMP_pack_array_5_SHORT		_XCALABLEMP_SM_PACK_ARRAY_5(short)
extern void _XCALABLEMP_pack_array_5_UNSIGNED_SHORT	_XCALABLEMP_SM_PACK_ARRAY_5(unsigned short)
extern void _XCALABLEMP_pack_array_5_INT		_XCALABLEMP_SM_PACK_ARRAY_5(int)
extern void _XCALABLEMP_pack_array_5_UNSIGNED_INT	_XCALABLEMP_SM_PACK_ARRAY_5(unsigned int)
extern void _XCALABLEMP_pack_array_5_LONG		_XCALABLEMP_SM_PACK_ARRAY_5(long)
extern void _XCALABLEMP_pack_array_5_UNSIGNED_LONG	_XCALABLEMP_SM_PACK_ARRAY_5(unsigned long)
extern void _XCALABLEMP_pack_array_5_LONG_LONG		_XCALABLEMP_SM_PACK_ARRAY_5(long long)
extern void _XCALABLEMP_pack_array_5_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_PACK_ARRAY_5(unsigned long long)
extern void _XCALABLEMP_pack_array_5_FLOAT		_XCALABLEMP_SM_PACK_ARRAY_5(float)
extern void _XCALABLEMP_pack_array_5_DOUBLE		_XCALABLEMP_SM_PACK_ARRAY_5(double)
extern void _XCALABLEMP_pack_array_5_LONG_DOUBLE	_XCALABLEMP_SM_PACK_ARRAY_5(long double)

// --- dimension 6
#define _XCALABLEMP_SM_PACK_ARRAY_6(_type) \
(_type *buf_addr, const _type *const src_addr, \
 int src_lower0, int src_upper0, int src_stride0, const unsigned long long src_dim_acc0, \
 int src_lower1, int src_upper1, int src_stride1, const unsigned long long src_dim_acc1, \
 int src_lower2, int src_upper2, int src_stride2, const unsigned long long src_dim_acc2, \
 int src_lower3, int src_upper3, int src_stride3, const unsigned long long src_dim_acc3, \
 int src_lower4, int src_upper4, int src_stride4, const unsigned long long src_dim_acc4, \
 int src_lower5, int src_upper5, int src_stride5);

extern void _XCALABLEMP_pack_array_6_BOOL		_XCALABLEMP_SM_PACK_ARRAY_6(_Bool)
extern void _XCALABLEMP_pack_array_6_CHAR		_XCALABLEMP_SM_PACK_ARRAY_6(char)
extern void _XCALABLEMP_pack_array_6_UNSIGNED_CHAR	_XCALABLEMP_SM_PACK_ARRAY_6(unsigned char)
extern void _XCALABLEMP_pack_array_6_SHORT		_XCALABLEMP_SM_PACK_ARRAY_6(short)
extern void _XCALABLEMP_pack_array_6_UNSIGNED_SHORT	_XCALABLEMP_SM_PACK_ARRAY_6(unsigned short)
extern void _XCALABLEMP_pack_array_6_INT		_XCALABLEMP_SM_PACK_ARRAY_6(int)
extern void _XCALABLEMP_pack_array_6_UNSIGNED_INT	_XCALABLEMP_SM_PACK_ARRAY_6(unsigned int)
extern void _XCALABLEMP_pack_array_6_LONG		_XCALABLEMP_SM_PACK_ARRAY_6(long)
extern void _XCALABLEMP_pack_array_6_UNSIGNED_LONG	_XCALABLEMP_SM_PACK_ARRAY_6(unsigned long)
extern void _XCALABLEMP_pack_array_6_LONG_LONG		_XCALABLEMP_SM_PACK_ARRAY_6(long long)
extern void _XCALABLEMP_pack_array_6_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_PACK_ARRAY_6(unsigned long long)
extern void _XCALABLEMP_pack_array_6_FLOAT		_XCALABLEMP_SM_PACK_ARRAY_6(float)
extern void _XCALABLEMP_pack_array_6_DOUBLE		_XCALABLEMP_SM_PACK_ARRAY_6(double)
extern void _XCALABLEMP_pack_array_6_LONG_DOUBLE	_XCALABLEMP_SM_PACK_ARRAY_6(long double)

// --- dimension 7
#define _XCALABLEMP_SM_PACK_ARRAY_7(_type) \
(_type *buf_addr, const _type *const src_addr, \
 int src_lower0, int src_upper0, int src_stride0, const unsigned long long src_dim_acc0, \
 int src_lower1, int src_upper1, int src_stride1, const unsigned long long src_dim_acc1, \
 int src_lower2, int src_upper2, int src_stride2, const unsigned long long src_dim_acc2, \
 int src_lower3, int src_upper3, int src_stride3, const unsigned long long src_dim_acc3, \
 int src_lower4, int src_upper4, int src_stride4, const unsigned long long src_dim_acc4, \
 int src_lower5, int src_upper5, int src_stride5, const unsigned long long src_dim_acc5, \
 int src_lower6, int src_upper6, int src_stride6);

extern void _XCALABLEMP_pack_array_7_BOOL		_XCALABLEMP_SM_PACK_ARRAY_7(_Bool)
extern void _XCALABLEMP_pack_array_7_CHAR		_XCALABLEMP_SM_PACK_ARRAY_7(char)
extern void _XCALABLEMP_pack_array_7_UNSIGNED_CHAR	_XCALABLEMP_SM_PACK_ARRAY_7(unsigned char)
extern void _XCALABLEMP_pack_array_7_SHORT		_XCALABLEMP_SM_PACK_ARRAY_7(short)
extern void _XCALABLEMP_pack_array_7_UNSIGNED_SHORT	_XCALABLEMP_SM_PACK_ARRAY_7(unsigned short)
extern void _XCALABLEMP_pack_array_7_INT		_XCALABLEMP_SM_PACK_ARRAY_7(int)
extern void _XCALABLEMP_pack_array_7_UNSIGNED_INT	_XCALABLEMP_SM_PACK_ARRAY_7(unsigned int)
extern void _XCALABLEMP_pack_array_7_LONG		_XCALABLEMP_SM_PACK_ARRAY_7(long)
extern void _XCALABLEMP_pack_array_7_UNSIGNED_LONG	_XCALABLEMP_SM_PACK_ARRAY_7(unsigned long)
extern void _XCALABLEMP_pack_array_7_LONG_LONG		_XCALABLEMP_SM_PACK_ARRAY_7(long long)
extern void _XCALABLEMP_pack_array_7_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_PACK_ARRAY_7(unsigned long long)
extern void _XCALABLEMP_pack_array_7_FLOAT		_XCALABLEMP_SM_PACK_ARRAY_7(float)
extern void _XCALABLEMP_pack_array_7_DOUBLE		_XCALABLEMP_SM_PACK_ARRAY_7(double)
extern void _XCALABLEMP_pack_array_7_LONG_DOUBLE	_XCALABLEMP_SM_PACK_ARRAY_7(long double)


// ----- unpack array
// --- dimension 1
#define _XCALABLEMP_SM_UNPACK_ARRAY_1(_type) \
(_type *dst_addr, _type *buf_addr, \
 int dst_lower, int dst_upper, int dst_stride);

extern void _XCALABLEMP_unpack_array_1_BOOL			_XCALABLEMP_SM_UNPACK_ARRAY_1(_Bool)
extern void _XCALABLEMP_unpack_array_1_CHAR			_XCALABLEMP_SM_UNPACK_ARRAY_1(char)
extern void _XCALABLEMP_unpack_array_1_UNSIGNED_CHAR		_XCALABLEMP_SM_UNPACK_ARRAY_1(unsigned char)
extern void _XCALABLEMP_unpack_array_1_SHORT			_XCALABLEMP_SM_UNPACK_ARRAY_1(short)
extern void _XCALABLEMP_unpack_array_1_UNSIGNED_SHORT		_XCALABLEMP_SM_UNPACK_ARRAY_1(unsigned short)
extern void _XCALABLEMP_unpack_array_1_INT			_XCALABLEMP_SM_UNPACK_ARRAY_1(int)
extern void _XCALABLEMP_unpack_array_1_UNSIGNED_INT		_XCALABLEMP_SM_UNPACK_ARRAY_1(unsigned int)
extern void _XCALABLEMP_unpack_array_1_LONG			_XCALABLEMP_SM_UNPACK_ARRAY_1(long)
extern void _XCALABLEMP_unpack_array_1_UNSIGNED_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_1(unsigned long)
extern void _XCALABLEMP_unpack_array_1_LONG_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_1(long long)
extern void _XCALABLEMP_unpack_array_1_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_UNPACK_ARRAY_1(unsigned long long)
extern void _XCALABLEMP_unpack_array_1_FLOAT			_XCALABLEMP_SM_UNPACK_ARRAY_1(float)
extern void _XCALABLEMP_unpack_array_1_DOUBLE			_XCALABLEMP_SM_UNPACK_ARRAY_1(double)
extern void _XCALABLEMP_unpack_array_1_LONG_DOUBLE		_XCALABLEMP_SM_UNPACK_ARRAY_1(long double)

// --- dimension 2
#define _XCALABLEMP_SM_UNPACK_ARRAY_2(_type) \
(_type *dst_addr, _type *buf_addr, \
 int dst_lower0, int dst_upper0, int dst_stride0, unsigned long long dst_dim_acc0, \
 int dst_lower1, int dst_upper1, int dst_stride1);

extern void _XCALABLEMP_unpack_array_2_BOOL			_XCALABLEMP_SM_UNPACK_ARRAY_2(_Bool)
extern void _XCALABLEMP_unpack_array_2_CHAR			_XCALABLEMP_SM_UNPACK_ARRAY_2(char)
extern void _XCALABLEMP_unpack_array_2_UNSIGNED_CHAR		_XCALABLEMP_SM_UNPACK_ARRAY_2(unsigned char)
extern void _XCALABLEMP_unpack_array_2_SHORT			_XCALABLEMP_SM_UNPACK_ARRAY_1(short)
extern void _XCALABLEMP_unpack_array_2_UNSIGNED_SHORT		_XCALABLEMP_SM_UNPACK_ARRAY_2(unsigned short)
extern void _XCALABLEMP_unpack_array_2_INT			_XCALABLEMP_SM_UNPACK_ARRAY_1(int)
extern void _XCALABLEMP_unpack_array_2_UNSIGNED_INT		_XCALABLEMP_SM_UNPACK_ARRAY_2(unsigned int)
extern void _XCALABLEMP_unpack_array_2_LONG			_XCALABLEMP_SM_UNPACK_ARRAY_1(long)
extern void _XCALABLEMP_unpack_array_2_UNSIGNED_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_2(unsigned long)
extern void _XCALABLEMP_unpack_array_2_LONG_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_2(long long)
extern void _XCALABLEMP_unpack_array_2_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_UNPACK_ARRAY_2(unsigned long long)
extern void _XCALABLEMP_unpack_array_2_FLOAT			_XCALABLEMP_SM_UNPACK_ARRAY_2(float)
extern void _XCALABLEMP_unpack_array_2_DOUBLE			_XCALABLEMP_SM_UNPACK_ARRAY_2(double)
extern void _XCALABLEMP_unpack_array_2_LONG_DOUBLE		_XCALABLEMP_SM_UNPACK_ARRAY_2(long double)

// --- dimension 3
#define _XCALABLEMP_SM_UNPACK_ARRAY_3(_type) \
(_type *const dst_addr, _type *buf_addr, \
 int dst_lower0, int dst_upper0, int dst_stride0, const unsigned long long dst_dim_acc0, \
 int dst_lower1, int dst_upper1, int dst_stride1, const unsigned long long dst_dim_acc1, \
 int dst_lower2, int dst_upper2, int dst_stride2);

extern void _XCALABLEMP_unpack_array_3_BOOL			_XCALABLEMP_SM_UNPACK_ARRAY_3(_Bool)
extern void _XCALABLEMP_unpack_array_3_CHAR			_XCALABLEMP_SM_UNPACK_ARRAY_3(char)
extern void _XCALABLEMP_unpack_array_3_UNSIGNED_CHAR		_XCALABLEMP_SM_UNPACK_ARRAY_3(unsigned char)
extern void _XCALABLEMP_unpack_array_3_SHORT			_XCALABLEMP_SM_UNPACK_ARRAY_3(short)
extern void _XCALABLEMP_unpack_array_3_UNSIGNED_SHORT		_XCALABLEMP_SM_UNPACK_ARRAY_3(unsigned short)
extern void _XCALABLEMP_unpack_array_3_INT			_XCALABLEMP_SM_UNPACK_ARRAY_3(int)
extern void _XCALABLEMP_unpack_array_3_UNSIGNED_INT		_XCALABLEMP_SM_UNPACK_ARRAY_3(unsigned int)
extern void _XCALABLEMP_unpack_array_3_LONG			_XCALABLEMP_SM_UNPACK_ARRAY_3(long)
extern void _XCALABLEMP_unpack_array_3_UNSIGNED_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_3(unsigned long)
extern void _XCALABLEMP_unpack_array_3_LONG_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_3(long long)
extern void _XCALABLEMP_unpack_array_3_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_UNPACK_ARRAY_3(unsigned long long)
extern void _XCALABLEMP_unpack_array_3_FLOAT			_XCALABLEMP_SM_UNPACK_ARRAY_3(float)
extern void _XCALABLEMP_unpack_array_3_DOUBLE			_XCALABLEMP_SM_UNPACK_ARRAY_3(double)
extern void _XCALABLEMP_unpack_array_3_LONG_DOUBLE		_XCALABLEMP_SM_UNPACK_ARRAY_3(long double)

// --- dimension 4
#define _XCALABLEMP_SM_UNPACK_ARRAY_4(_type) \
(_type *const dst_addr, _type *buf_addr, \
 int dst_lower0, int dst_upper0, int dst_stride0, const unsigned long long dst_dim_acc0, \
 int dst_lower1, int dst_upper1, int dst_stride1, const unsigned long long dst_dim_acc1, \
 int dst_lower2, int dst_upper2, int dst_stride2, const unsigned long long dst_dim_acc2, \
 int dst_lower3, int dst_upper3, int dst_stride3);

extern void _XCALABLEMP_unpack_array_4_BOOL			_XCALABLEMP_SM_UNPACK_ARRAY_4(_Bool)
extern void _XCALABLEMP_unpack_array_4_CHAR			_XCALABLEMP_SM_UNPACK_ARRAY_4(char)
extern void _XCALABLEMP_unpack_array_4_UNSIGNED_CHAR		_XCALABLEMP_SM_UNPACK_ARRAY_4(unsigned char)
extern void _XCALABLEMP_unpack_array_4_SHORT			_XCALABLEMP_SM_UNPACK_ARRAY_4(short)
extern void _XCALABLEMP_unpack_array_4_UNSIGNED_SHORT		_XCALABLEMP_SM_UNPACK_ARRAY_4(unsigned short)
extern void _XCALABLEMP_unpack_array_4_INT			_XCALABLEMP_SM_UNPACK_ARRAY_4(int)
extern void _XCALABLEMP_unpack_array_4_UNSIGNED_INT		_XCALABLEMP_SM_UNPACK_ARRAY_4(unsigned int)
extern void _XCALABLEMP_unpack_array_4_LONG			_XCALABLEMP_SM_UNPACK_ARRAY_4(long)
extern void _XCALABLEMP_unpack_array_4_UNSIGNED_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_4(unsigned long)
extern void _XCALABLEMP_unpack_array_4_LONG_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_4(long long)
extern void _XCALABLEMP_unpack_array_4_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_UNPACK_ARRAY_4(unsigned long long)
extern void _XCALABLEMP_unpack_array_4_FLOAT			_XCALABLEMP_SM_UNPACK_ARRAY_4(float)
extern void _XCALABLEMP_unpack_array_4_DOUBLE			_XCALABLEMP_SM_UNPACK_ARRAY_4(double)
extern void _XCALABLEMP_unpack_array_4_LONG_DOUBLE		_XCALABLEMP_SM_UNPACK_ARRAY_4(long double)

// --- dimension 5
#define _XCALABLEMP_SM_UNPACK_ARRAY_5(_type) \
(_type *const dst_addr, _type *buf_addr, \
 int dst_lower0, int dst_upper0, int dst_stride0, const unsigned long long dst_dim_acc0, \
 int dst_lower1, int dst_upper1, int dst_stride1, const unsigned long long dst_dim_acc1, \
 int dst_lower2, int dst_upper2, int dst_stride2, const unsigned long long dst_dim_acc2, \
 int dst_lower3, int dst_upper3, int dst_stride3, const unsigned long long dst_dim_acc3, \
 int dst_lower4, int dst_upper4, int dst_stride4);

extern void _XCALABLEMP_unpack_array_5_BOOL			_XCALABLEMP_SM_UNPACK_ARRAY_5(_Bool)
extern void _XCALABLEMP_unpack_array_5_CHAR			_XCALABLEMP_SM_UNPACK_ARRAY_5(char)
extern void _XCALABLEMP_unpack_array_5_UNSIGNED_CHAR		_XCALABLEMP_SM_UNPACK_ARRAY_5(unsigned char)
extern void _XCALABLEMP_unpack_array_5_SHORT			_XCALABLEMP_SM_UNPACK_ARRAY_5(short)
extern void _XCALABLEMP_unpack_array_5_UNSIGNED_SHORT		_XCALABLEMP_SM_UNPACK_ARRAY_5(unsigned short)
extern void _XCALABLEMP_unpack_array_5_INT			_XCALABLEMP_SM_UNPACK_ARRAY_5(int)
extern void _XCALABLEMP_unpack_array_5_UNSIGNED_INT		_XCALABLEMP_SM_UNPACK_ARRAY_5(unsigned int)
extern void _XCALABLEMP_unpack_array_5_LONG			_XCALABLEMP_SM_UNPACK_ARRAY_5(long)
extern void _XCALABLEMP_unpack_array_5_UNSIGNED_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_5(unsigned long)
extern void _XCALABLEMP_unpack_array_5_LONG_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_5(long long)
extern void _XCALABLEMP_unpack_array_5_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_UNPACK_ARRAY_5(unsigned long long)
extern void _XCALABLEMP_unpack_array_5_FLOAT			_XCALABLEMP_SM_UNPACK_ARRAY_5(float)
extern void _XCALABLEMP_unpack_array_5_DOUBLE			_XCALABLEMP_SM_UNPACK_ARRAY_5(double)
extern void _XCALABLEMP_unpack_array_5_LONG_DOUBLE		_XCALABLEMP_SM_UNPACK_ARRAY_5(long double)

// --- dimension 6
#define _XCALABLEMP_SM_UNPACK_ARRAY_6(_type) \
(_type *const dst_addr, _type *buf_addr, \
 int dst_lower0, int dst_upper0, int dst_stride0, const unsigned long long dst_dim_acc0, \
 int dst_lower1, int dst_upper1, int dst_stride1, const unsigned long long dst_dim_acc1, \
 int dst_lower2, int dst_upper2, int dst_stride2, const unsigned long long dst_dim_acc2, \
 int dst_lower3, int dst_upper3, int dst_stride3, const unsigned long long dst_dim_acc3, \
 int dst_lower4, int dst_upper4, int dst_stride4, const unsigned long long dst_dim_acc4, \
 int dst_lower5, int dst_upper5, int dst_stride5);

extern void _XCALABLEMP_unpack_array_6_BOOL			_XCALABLEMP_SM_UNPACK_ARRAY_6(_Bool)
extern void _XCALABLEMP_unpack_array_6_CHAR			_XCALABLEMP_SM_UNPACK_ARRAY_6(char)
extern void _XCALABLEMP_unpack_array_6_UNSIGNED_CHAR		_XCALABLEMP_SM_UNPACK_ARRAY_6(unsigned char)
extern void _XCALABLEMP_unpack_array_6_SHORT			_XCALABLEMP_SM_UNPACK_ARRAY_6(short)
extern void _XCALABLEMP_unpack_array_6_UNSIGNED_SHORT		_XCALABLEMP_SM_UNPACK_ARRAY_6(unsigned short)
extern void _XCALABLEMP_unpack_array_6_INT			_XCALABLEMP_SM_UNPACK_ARRAY_6(int)
extern void _XCALABLEMP_unpack_array_6_UNSIGNED_INT		_XCALABLEMP_SM_UNPACK_ARRAY_6(unsigned int)
extern void _XCALABLEMP_unpack_array_6_LONG			_XCALABLEMP_SM_UNPACK_ARRAY_6(long)
extern void _XCALABLEMP_unpack_array_6_UNSIGNED_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_6(unsigned long)
extern void _XCALABLEMP_unpack_array_6_LONG_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_6(long long)
extern void _XCALABLEMP_unpack_array_6_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_UNPACK_ARRAY_6(unsigned long long)
extern void _XCALABLEMP_unpack_array_6_FLOAT			_XCALABLEMP_SM_UNPACK_ARRAY_6(float)
extern void _XCALABLEMP_unpack_array_6_DOUBLE			_XCALABLEMP_SM_UNPACK_ARRAY_6(double)
extern void _XCALABLEMP_unpack_array_6_LONG_DOUBLE		_XCALABLEMP_SM_UNPACK_ARRAY_6(long double)

// --- dimension 7
#define _XCALABLEMP_SM_UNPACK_ARRAY_7(_type) \
(_type *const dst_addr, _type *buf_addr, \
 int dst_lower0, int dst_upper0, int dst_stride0, const unsigned long long dst_dim_acc0, \
 int dst_lower1, int dst_upper1, int dst_stride1, const unsigned long long dst_dim_acc1, \
 int dst_lower2, int dst_upper2, int dst_stride2, const unsigned long long dst_dim_acc2, \
 int dst_lower3, int dst_upper3, int dst_stride3, const unsigned long long dst_dim_acc3, \
 int dst_lower4, int dst_upper4, int dst_stride4, const unsigned long long dst_dim_acc4, \
 int dst_lower5, int dst_upper5, int dst_stride5, const unsigned long long dst_dim_acc5, \
 int dst_lower6, int dst_upper6, int dst_stride6);

extern void _XCALABLEMP_unpack_array_7_BOOL			_XCALABLEMP_SM_UNPACK_ARRAY_7(_Bool)
extern void _XCALABLEMP_unpack_array_7_CHAR			_XCALABLEMP_SM_UNPACK_ARRAY_7(char)
extern void _XCALABLEMP_unpack_array_7_UNSIGNED_CHAR		_XCALABLEMP_SM_UNPACK_ARRAY_7(unsigned char)
extern void _XCALABLEMP_unpack_array_7_SHORT			_XCALABLEMP_SM_UNPACK_ARRAY_7(short)
extern void _XCALABLEMP_unpack_array_7_UNSIGNED_SHORT		_XCALABLEMP_SM_UNPACK_ARRAY_7(unsigned short)
extern void _XCALABLEMP_unpack_array_7_INT			_XCALABLEMP_SM_UNPACK_ARRAY_7(int)
extern void _XCALABLEMP_unpack_array_7_UNSIGNED_INT		_XCALABLEMP_SM_UNPACK_ARRAY_7(unsigned int)
extern void _XCALABLEMP_unpack_array_7_LONG			_XCALABLEMP_SM_UNPACK_ARRAY_7(long)
extern void _XCALABLEMP_unpack_array_7_UNSIGNED_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_7(unsigned long)
extern void _XCALABLEMP_unpack_array_7_LONG_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_7(long long)
extern void _XCALABLEMP_unpack_array_7_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_UNPACK_ARRAY_7(unsigned long long)
extern void _XCALABLEMP_unpack_array_7_FLOAT			_XCALABLEMP_SM_UNPACK_ARRAY_7(float)
extern void _XCALABLEMP_unpack_array_7_DOUBLE			_XCALABLEMP_SM_UNPACK_ARRAY_7(double)
extern void _XCALABLEMP_unpack_array_7_LONG_DOUBLE		_XCALABLEMP_SM_UNPACK_ARRAY_7(long double)

// pack/unpack shadow
extern void _XCALABLEMP_pack_shadow_buffer(void *buffer, void *src,
                                           int array_type, int array_dim, int *l, int *u, int *s, unsigned long long *d);
extern void _XCALABLEMP_unpack_shadow_buffer(void *dst, void *buffer,
                                             int array_type, int array_dim, int *l, int *u, int *s, unsigned long long *d);

#endif // _XCALABLEMP_ARRAY_SECTION
