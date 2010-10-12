#ifndef _XCALABLEMP_ARRAY_SECTION
#define _XCALABLEMP_ARRAY_SECTION

// ----- pack array
#define _XCALABLEMP_SM_PACK_ARRAY_1(_type) \
(_type *buf_addr, _type *src_addr, \
 int src_lower, int src_upper, int src_stride);

#define _XCALABLEMP_SM_PACK_ARRAY_N(_type) \
(_type *buf_addr, _type *src_addr, \
 int *l, int *u, int *s, unsigned long long *d);

// --- dimension 1
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
extern void _XCALABLEMP_pack_array_2_BOOL		_XCALABLEMP_SM_PACK_ARRAY_N(_Bool)
extern void _XCALABLEMP_pack_array_2_CHAR		_XCALABLEMP_SM_PACK_ARRAY_N(char)
extern void _XCALABLEMP_pack_array_2_UNSIGNED_CHAR	_XCALABLEMP_SM_PACK_ARRAY_N(unsigned char)
extern void _XCALABLEMP_pack_array_2_SHORT		_XCALABLEMP_SM_PACK_ARRAY_N(short)
extern void _XCALABLEMP_pack_array_2_UNSIGNED_SHORT	_XCALABLEMP_SM_PACK_ARRAY_N(unsigned short)
extern void _XCALABLEMP_pack_array_2_INT		_XCALABLEMP_SM_PACK_ARRAY_N(int)
extern void _XCALABLEMP_pack_array_2_UNSIGNED_INT	_XCALABLEMP_SM_PACK_ARRAY_N(unsigned int)
extern void _XCALABLEMP_pack_array_2_LONG		_XCALABLEMP_SM_PACK_ARRAY_N(long)
extern void _XCALABLEMP_pack_array_2_UNSIGNED_LONG	_XCALABLEMP_SM_PACK_ARRAY_N(unsigned long)
extern void _XCALABLEMP_pack_array_2_LONG_LONG		_XCALABLEMP_SM_PACK_ARRAY_N(long long)
extern void _XCALABLEMP_pack_array_2_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_PACK_ARRAY_N(unsigned long long)
extern void _XCALABLEMP_pack_array_2_FLOAT		_XCALABLEMP_SM_PACK_ARRAY_N(float)
extern void _XCALABLEMP_pack_array_2_DOUBLE		_XCALABLEMP_SM_PACK_ARRAY_N(double)
extern void _XCALABLEMP_pack_array_2_LONG_DOUBLE	_XCALABLEMP_SM_PACK_ARRAY_N(long double)

// --- dimension 3
extern void _XCALABLEMP_pack_array_3_BOOL		_XCALABLEMP_SM_PACK_ARRAY_N(_Bool)
extern void _XCALABLEMP_pack_array_3_CHAR		_XCALABLEMP_SM_PACK_ARRAY_N(char)
extern void _XCALABLEMP_pack_array_3_UNSIGNED_CHAR	_XCALABLEMP_SM_PACK_ARRAY_N(unsigned char)
extern void _XCALABLEMP_pack_array_3_SHORT		_XCALABLEMP_SM_PACK_ARRAY_N(short)
extern void _XCALABLEMP_pack_array_3_UNSIGNED_SHORT	_XCALABLEMP_SM_PACK_ARRAY_N(unsigned short)
extern void _XCALABLEMP_pack_array_3_INT		_XCALABLEMP_SM_PACK_ARRAY_N(int)
extern void _XCALABLEMP_pack_array_3_UNSIGNED_INT	_XCALABLEMP_SM_PACK_ARRAY_N(unsigned int)
extern void _XCALABLEMP_pack_array_3_LONG		_XCALABLEMP_SM_PACK_ARRAY_N(long)
extern void _XCALABLEMP_pack_array_3_UNSIGNED_LONG	_XCALABLEMP_SM_PACK_ARRAY_N(unsigned long)
extern void _XCALABLEMP_pack_array_3_LONG_LONG		_XCALABLEMP_SM_PACK_ARRAY_N(long long)
extern void _XCALABLEMP_pack_array_3_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_PACK_ARRAY_N(unsigned long long)
extern void _XCALABLEMP_pack_array_3_FLOAT		_XCALABLEMP_SM_PACK_ARRAY_N(float)
extern void _XCALABLEMP_pack_array_3_DOUBLE		_XCALABLEMP_SM_PACK_ARRAY_N(double)
extern void _XCALABLEMP_pack_array_3_LONG_DOUBLE	_XCALABLEMP_SM_PACK_ARRAY_N(long double)

// --- dimension 4
extern void _XCALABLEMP_pack_array_4_BOOL		_XCALABLEMP_SM_PACK_ARRAY_N(_Bool)
extern void _XCALABLEMP_pack_array_4_CHAR		_XCALABLEMP_SM_PACK_ARRAY_N(char)
extern void _XCALABLEMP_pack_array_4_UNSIGNED_CHAR	_XCALABLEMP_SM_PACK_ARRAY_N(unsigned char)
extern void _XCALABLEMP_pack_array_4_SHORT		_XCALABLEMP_SM_PACK_ARRAY_N(short)
extern void _XCALABLEMP_pack_array_4_UNSIGNED_SHORT	_XCALABLEMP_SM_PACK_ARRAY_N(unsigned short)
extern void _XCALABLEMP_pack_array_4_INT		_XCALABLEMP_SM_PACK_ARRAY_N(int)
extern void _XCALABLEMP_pack_array_4_UNSIGNED_INT	_XCALABLEMP_SM_PACK_ARRAY_N(unsigned int)
extern void _XCALABLEMP_pack_array_4_LONG		_XCALABLEMP_SM_PACK_ARRAY_N(long)
extern void _XCALABLEMP_pack_array_4_UNSIGNED_LONG	_XCALABLEMP_SM_PACK_ARRAY_N(unsigned long)
extern void _XCALABLEMP_pack_array_4_LONG_LONG		_XCALABLEMP_SM_PACK_ARRAY_N(long long)
extern void _XCALABLEMP_pack_array_4_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_PACK_ARRAY_N(unsigned long long)
extern void _XCALABLEMP_pack_array_4_FLOAT		_XCALABLEMP_SM_PACK_ARRAY_N(float)
extern void _XCALABLEMP_pack_array_4_DOUBLE		_XCALABLEMP_SM_PACK_ARRAY_N(double)
extern void _XCALABLEMP_pack_array_4_LONG_DOUBLE	_XCALABLEMP_SM_PACK_ARRAY_N(long double)

// --- dimension 5
extern void _XCALABLEMP_pack_array_5_BOOL		_XCALABLEMP_SM_PACK_ARRAY_N(_Bool)
extern void _XCALABLEMP_pack_array_5_CHAR		_XCALABLEMP_SM_PACK_ARRAY_N(char)
extern void _XCALABLEMP_pack_array_5_UNSIGNED_CHAR	_XCALABLEMP_SM_PACK_ARRAY_N(unsigned char)
extern void _XCALABLEMP_pack_array_5_SHORT		_XCALABLEMP_SM_PACK_ARRAY_N(short)
extern void _XCALABLEMP_pack_array_5_UNSIGNED_SHORT	_XCALABLEMP_SM_PACK_ARRAY_N(unsigned short)
extern void _XCALABLEMP_pack_array_5_INT		_XCALABLEMP_SM_PACK_ARRAY_N(int)
extern void _XCALABLEMP_pack_array_5_UNSIGNED_INT	_XCALABLEMP_SM_PACK_ARRAY_N(unsigned int)
extern void _XCALABLEMP_pack_array_5_LONG		_XCALABLEMP_SM_PACK_ARRAY_N(long)
extern void _XCALABLEMP_pack_array_5_UNSIGNED_LONG	_XCALABLEMP_SM_PACK_ARRAY_N(unsigned long)
extern void _XCALABLEMP_pack_array_5_LONG_LONG		_XCALABLEMP_SM_PACK_ARRAY_N(long long)
extern void _XCALABLEMP_pack_array_5_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_PACK_ARRAY_N(unsigned long long)
extern void _XCALABLEMP_pack_array_5_FLOAT		_XCALABLEMP_SM_PACK_ARRAY_N(float)
extern void _XCALABLEMP_pack_array_5_DOUBLE		_XCALABLEMP_SM_PACK_ARRAY_N(double)
extern void _XCALABLEMP_pack_array_5_LONG_DOUBLE	_XCALABLEMP_SM_PACK_ARRAY_N(long double)

// --- dimension 6
extern void _XCALABLEMP_pack_array_6_BOOL		_XCALABLEMP_SM_PACK_ARRAY_N(_Bool)
extern void _XCALABLEMP_pack_array_6_CHAR		_XCALABLEMP_SM_PACK_ARRAY_N(char)
extern void _XCALABLEMP_pack_array_6_UNSIGNED_CHAR	_XCALABLEMP_SM_PACK_ARRAY_N(unsigned char)
extern void _XCALABLEMP_pack_array_6_SHORT		_XCALABLEMP_SM_PACK_ARRAY_N(short)
extern void _XCALABLEMP_pack_array_6_UNSIGNED_SHORT	_XCALABLEMP_SM_PACK_ARRAY_N(unsigned short)
extern void _XCALABLEMP_pack_array_6_INT		_XCALABLEMP_SM_PACK_ARRAY_N(int)
extern void _XCALABLEMP_pack_array_6_UNSIGNED_INT	_XCALABLEMP_SM_PACK_ARRAY_N(unsigned int)
extern void _XCALABLEMP_pack_array_6_LONG		_XCALABLEMP_SM_PACK_ARRAY_N(long)
extern void _XCALABLEMP_pack_array_6_UNSIGNED_LONG	_XCALABLEMP_SM_PACK_ARRAY_N(unsigned long)
extern void _XCALABLEMP_pack_array_6_LONG_LONG		_XCALABLEMP_SM_PACK_ARRAY_N(long long)
extern void _XCALABLEMP_pack_array_6_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_PACK_ARRAY_N(unsigned long long)
extern void _XCALABLEMP_pack_array_6_FLOAT		_XCALABLEMP_SM_PACK_ARRAY_N(float)
extern void _XCALABLEMP_pack_array_6_DOUBLE		_XCALABLEMP_SM_PACK_ARRAY_N(double)
extern void _XCALABLEMP_pack_array_6_LONG_DOUBLE	_XCALABLEMP_SM_PACK_ARRAY_N(long double)

// --- dimension 7
extern void _XCALABLEMP_pack_array_7_BOOL		_XCALABLEMP_SM_PACK_ARRAY_N(_Bool)
extern void _XCALABLEMP_pack_array_7_CHAR		_XCALABLEMP_SM_PACK_ARRAY_N(char)
extern void _XCALABLEMP_pack_array_7_UNSIGNED_CHAR	_XCALABLEMP_SM_PACK_ARRAY_N(unsigned char)
extern void _XCALABLEMP_pack_array_7_SHORT		_XCALABLEMP_SM_PACK_ARRAY_N(short)
extern void _XCALABLEMP_pack_array_7_UNSIGNED_SHORT	_XCALABLEMP_SM_PACK_ARRAY_N(unsigned short)
extern void _XCALABLEMP_pack_array_7_INT		_XCALABLEMP_SM_PACK_ARRAY_N(int)
extern void _XCALABLEMP_pack_array_7_UNSIGNED_INT	_XCALABLEMP_SM_PACK_ARRAY_N(unsigned int)
extern void _XCALABLEMP_pack_array_7_LONG		_XCALABLEMP_SM_PACK_ARRAY_N(long)
extern void _XCALABLEMP_pack_array_7_UNSIGNED_LONG	_XCALABLEMP_SM_PACK_ARRAY_N(unsigned long)
extern void _XCALABLEMP_pack_array_7_LONG_LONG		_XCALABLEMP_SM_PACK_ARRAY_N(long long)
extern void _XCALABLEMP_pack_array_7_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_PACK_ARRAY_N(unsigned long long)
extern void _XCALABLEMP_pack_array_7_FLOAT		_XCALABLEMP_SM_PACK_ARRAY_N(float)
extern void _XCALABLEMP_pack_array_7_DOUBLE		_XCALABLEMP_SM_PACK_ARRAY_N(double)
extern void _XCALABLEMP_pack_array_7_LONG_DOUBLE	_XCALABLEMP_SM_PACK_ARRAY_N(long double)


// ----- unpack array
#define _XCALABLEMP_SM_UNPACK_ARRAY_1(_type) \
(_type *dst_addr, _type *buf_addr, \
 int dst_lower, int dst_upper, int dst_stride);

#define _XCALABLEMP_SM_UNPACK_ARRAY_N(_type) \
(_type *dst_addr, _type *buf_addr, \
 int *l, int *u, int *s, unsigned long long *d);

// --- dimension 1
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
extern void _XCALABLEMP_unpack_array_2_BOOL			_XCALABLEMP_SM_UNPACK_ARRAY_N(_Bool)
extern void _XCALABLEMP_unpack_array_2_CHAR			_XCALABLEMP_SM_UNPACK_ARRAY_N(char)
extern void _XCALABLEMP_unpack_array_2_UNSIGNED_CHAR		_XCALABLEMP_SM_UNPACK_ARRAY_N(unsigned char)
extern void _XCALABLEMP_unpack_array_2_SHORT			_XCALABLEMP_SM_UNPACK_ARRAY_N(short)
extern void _XCALABLEMP_unpack_array_2_UNSIGNED_SHORT		_XCALABLEMP_SM_UNPACK_ARRAY_N(unsigned short)
extern void _XCALABLEMP_unpack_array_2_INT			_XCALABLEMP_SM_UNPACK_ARRAY_N(int)
extern void _XCALABLEMP_unpack_array_2_UNSIGNED_INT		_XCALABLEMP_SM_UNPACK_ARRAY_N(unsigned int)
extern void _XCALABLEMP_unpack_array_2_LONG			_XCALABLEMP_SM_UNPACK_ARRAY_N(long)
extern void _XCALABLEMP_unpack_array_2_UNSIGNED_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_N(unsigned long)
extern void _XCALABLEMP_unpack_array_2_LONG_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_N(long long)
extern void _XCALABLEMP_unpack_array_2_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_UNPACK_ARRAY_N(unsigned long long)
extern void _XCALABLEMP_unpack_array_2_FLOAT			_XCALABLEMP_SM_UNPACK_ARRAY_N(float)
extern void _XCALABLEMP_unpack_array_2_DOUBLE			_XCALABLEMP_SM_UNPACK_ARRAY_N(double)
extern void _XCALABLEMP_unpack_array_2_LONG_DOUBLE		_XCALABLEMP_SM_UNPACK_ARRAY_N(long double)

// --- dimension 3
extern void _XCALABLEMP_unpack_array_3_BOOL			_XCALABLEMP_SM_UNPACK_ARRAY_N(_Bool)
extern void _XCALABLEMP_unpack_array_3_CHAR			_XCALABLEMP_SM_UNPACK_ARRAY_N(char)
extern void _XCALABLEMP_unpack_array_3_UNSIGNED_CHAR		_XCALABLEMP_SM_UNPACK_ARRAY_N(unsigned char)
extern void _XCALABLEMP_unpack_array_3_SHORT			_XCALABLEMP_SM_UNPACK_ARRAY_N(short)
extern void _XCALABLEMP_unpack_array_3_UNSIGNED_SHORT		_XCALABLEMP_SM_UNPACK_ARRAY_N(unsigned short)
extern void _XCALABLEMP_unpack_array_3_INT			_XCALABLEMP_SM_UNPACK_ARRAY_N(int)
extern void _XCALABLEMP_unpack_array_3_UNSIGNED_INT		_XCALABLEMP_SM_UNPACK_ARRAY_N(unsigned int)
extern void _XCALABLEMP_unpack_array_3_LONG			_XCALABLEMP_SM_UNPACK_ARRAY_N(long)
extern void _XCALABLEMP_unpack_array_3_UNSIGNED_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_N(unsigned long)
extern void _XCALABLEMP_unpack_array_3_LONG_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_N(long long)
extern void _XCALABLEMP_unpack_array_3_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_UNPACK_ARRAY_N(unsigned long long)
extern void _XCALABLEMP_unpack_array_3_FLOAT			_XCALABLEMP_SM_UNPACK_ARRAY_N(float)
extern void _XCALABLEMP_unpack_array_3_DOUBLE			_XCALABLEMP_SM_UNPACK_ARRAY_N(double)
extern void _XCALABLEMP_unpack_array_3_LONG_DOUBLE		_XCALABLEMP_SM_UNPACK_ARRAY_N(long double)

// --- dimension 4
extern void _XCALABLEMP_unpack_array_4_BOOL			_XCALABLEMP_SM_UNPACK_ARRAY_N(_Bool)
extern void _XCALABLEMP_unpack_array_4_CHAR			_XCALABLEMP_SM_UNPACK_ARRAY_N(char)
extern void _XCALABLEMP_unpack_array_4_UNSIGNED_CHAR		_XCALABLEMP_SM_UNPACK_ARRAY_N(unsigned char)
extern void _XCALABLEMP_unpack_array_4_SHORT			_XCALABLEMP_SM_UNPACK_ARRAY_N(short)
extern void _XCALABLEMP_unpack_array_4_UNSIGNED_SHORT		_XCALABLEMP_SM_UNPACK_ARRAY_N(unsigned short)
extern void _XCALABLEMP_unpack_array_4_INT			_XCALABLEMP_SM_UNPACK_ARRAY_N(int)
extern void _XCALABLEMP_unpack_array_4_UNSIGNED_INT		_XCALABLEMP_SM_UNPACK_ARRAY_N(unsigned int)
extern void _XCALABLEMP_unpack_array_4_LONG			_XCALABLEMP_SM_UNPACK_ARRAY_N(long)
extern void _XCALABLEMP_unpack_array_4_UNSIGNED_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_N(unsigned long)
extern void _XCALABLEMP_unpack_array_4_LONG_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_N(long long)
extern void _XCALABLEMP_unpack_array_4_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_UNPACK_ARRAY_N(unsigned long long)
extern void _XCALABLEMP_unpack_array_4_FLOAT			_XCALABLEMP_SM_UNPACK_ARRAY_N(float)
extern void _XCALABLEMP_unpack_array_4_DOUBLE			_XCALABLEMP_SM_UNPACK_ARRAY_N(double)
extern void _XCALABLEMP_unpack_array_4_LONG_DOUBLE		_XCALABLEMP_SM_UNPACK_ARRAY_N(long double)

// --- dimension 5
extern void _XCALABLEMP_unpack_array_5_BOOL			_XCALABLEMP_SM_UNPACK_ARRAY_N(_Bool)
extern void _XCALABLEMP_unpack_array_5_CHAR			_XCALABLEMP_SM_UNPACK_ARRAY_N(char)
extern void _XCALABLEMP_unpack_array_5_UNSIGNED_CHAR		_XCALABLEMP_SM_UNPACK_ARRAY_N(unsigned char)
extern void _XCALABLEMP_unpack_array_5_SHORT			_XCALABLEMP_SM_UNPACK_ARRAY_N(short)
extern void _XCALABLEMP_unpack_array_5_UNSIGNED_SHORT		_XCALABLEMP_SM_UNPACK_ARRAY_N(unsigned short)
extern void _XCALABLEMP_unpack_array_5_INT			_XCALABLEMP_SM_UNPACK_ARRAY_N(int)
extern void _XCALABLEMP_unpack_array_5_UNSIGNED_INT		_XCALABLEMP_SM_UNPACK_ARRAY_N(unsigned int)
extern void _XCALABLEMP_unpack_array_5_LONG			_XCALABLEMP_SM_UNPACK_ARRAY_N(long)
extern void _XCALABLEMP_unpack_array_5_UNSIGNED_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_N(unsigned long)
extern void _XCALABLEMP_unpack_array_5_LONG_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_N(long long)
extern void _XCALABLEMP_unpack_array_5_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_UNPACK_ARRAY_N(unsigned long long)
extern void _XCALABLEMP_unpack_array_5_FLOAT			_XCALABLEMP_SM_UNPACK_ARRAY_N(float)
extern void _XCALABLEMP_unpack_array_5_DOUBLE			_XCALABLEMP_SM_UNPACK_ARRAY_N(double)
extern void _XCALABLEMP_unpack_array_5_LONG_DOUBLE		_XCALABLEMP_SM_UNPACK_ARRAY_N(long double)

// --- dimension 6
extern void _XCALABLEMP_unpack_array_6_BOOL			_XCALABLEMP_SM_UNPACK_ARRAY_N(_Bool)
extern void _XCALABLEMP_unpack_array_6_CHAR			_XCALABLEMP_SM_UNPACK_ARRAY_N(char)
extern void _XCALABLEMP_unpack_array_6_UNSIGNED_CHAR		_XCALABLEMP_SM_UNPACK_ARRAY_N(unsigned char)
extern void _XCALABLEMP_unpack_array_6_SHORT			_XCALABLEMP_SM_UNPACK_ARRAY_N(short)
extern void _XCALABLEMP_unpack_array_6_UNSIGNED_SHORT		_XCALABLEMP_SM_UNPACK_ARRAY_N(unsigned short)
extern void _XCALABLEMP_unpack_array_6_INT			_XCALABLEMP_SM_UNPACK_ARRAY_N(int)
extern void _XCALABLEMP_unpack_array_6_UNSIGNED_INT		_XCALABLEMP_SM_UNPACK_ARRAY_N(unsigned int)
extern void _XCALABLEMP_unpack_array_6_LONG			_XCALABLEMP_SM_UNPACK_ARRAY_N(long)
extern void _XCALABLEMP_unpack_array_6_UNSIGNED_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_N(unsigned long)
extern void _XCALABLEMP_unpack_array_6_LONG_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_N(long long)
extern void _XCALABLEMP_unpack_array_6_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_UNPACK_ARRAY_N(unsigned long long)
extern void _XCALABLEMP_unpack_array_6_FLOAT			_XCALABLEMP_SM_UNPACK_ARRAY_N(float)
extern void _XCALABLEMP_unpack_array_6_DOUBLE			_XCALABLEMP_SM_UNPACK_ARRAY_N(double)
extern void _XCALABLEMP_unpack_array_6_LONG_DOUBLE		_XCALABLEMP_SM_UNPACK_ARRAY_N(long double)

// --- dimension 7
extern void _XCALABLEMP_unpack_array_7_BOOL			_XCALABLEMP_SM_UNPACK_ARRAY_N(_Bool)
extern void _XCALABLEMP_unpack_array_7_CHAR			_XCALABLEMP_SM_UNPACK_ARRAY_N(char)
extern void _XCALABLEMP_unpack_array_7_UNSIGNED_CHAR		_XCALABLEMP_SM_UNPACK_ARRAY_N(unsigned char)
extern void _XCALABLEMP_unpack_array_7_SHORT			_XCALABLEMP_SM_UNPACK_ARRAY_N(short)
extern void _XCALABLEMP_unpack_array_7_UNSIGNED_SHORT		_XCALABLEMP_SM_UNPACK_ARRAY_N(unsigned short)
extern void _XCALABLEMP_unpack_array_7_INT			_XCALABLEMP_SM_UNPACK_ARRAY_N(int)
extern void _XCALABLEMP_unpack_array_7_UNSIGNED_INT		_XCALABLEMP_SM_UNPACK_ARRAY_N(unsigned int)
extern void _XCALABLEMP_unpack_array_7_LONG			_XCALABLEMP_SM_UNPACK_ARRAY_N(long)
extern void _XCALABLEMP_unpack_array_7_UNSIGNED_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_N(unsigned long)
extern void _XCALABLEMP_unpack_array_7_LONG_LONG		_XCALABLEMP_SM_UNPACK_ARRAY_N(long long)
extern void _XCALABLEMP_unpack_array_7_UNSIGNED_LONG_LONG	_XCALABLEMP_SM_UNPACK_ARRAY_N(unsigned long long)
extern void _XCALABLEMP_unpack_array_7_FLOAT			_XCALABLEMP_SM_UNPACK_ARRAY_N(float)
extern void _XCALABLEMP_unpack_array_7_DOUBLE			_XCALABLEMP_SM_UNPACK_ARRAY_N(double)
extern void _XCALABLEMP_unpack_array_7_LONG_DOUBLE		_XCALABLEMP_SM_UNPACK_ARRAY_N(long double)

// pack/unpack shadow
extern void _XCALABLEMP_pack_shadow_buffer(void *buffer, void *src,
                                           int array_type, int array_dim, int *l, int *u, int *s, unsigned long long *d);
extern void _XCALABLEMP_unpack_shadow_buffer(void *dst, void *buffer,
                                             int array_type, int array_dim, int *l, int *u, int *s, unsigned long long *d);

#endif // _XCALABLEMP_ARRAY_SECTION
