#ifndef _ACC_PEZY_REDUCTION
#define  _ACC_PEZY_REDUCTION

#include "acc_pezy_util.hpp"

//#define DO_PREFETCH

typedef enum {
  PLUS   = 0,
  MUL    = 1,
  MAX    = 2,
  MIN    = 3,
  BITAND = 4,
  BITOR  = 5,
  BITXOR = 6,
  LOGAND = 7,
  LOGOR  = 8,
} _ACC_reduction_kind;

static inline
int _ACC_reduction_op(int a, int b, _ACC_reduction_kind kind)
{
  switch(kind){
  case PLUS: return a + b;
  case MUL: return a * b;
  case MAX: return (a > b)? a : b;
  case MIN: return (a < b)? a : b;
  case BITAND: return a & b;
  case BITOR: return a | b;
  case BITXOR: return a ^ b;
  case LOGAND: return a && b;
  case LOGOR: return a || b;
  default: return a;
  }
}

template<typename T>
static inline
T _ACC_reduction_op(T a, T b, _ACC_reduction_kind kind)
{
  switch(kind){
  case PLUS: return a + b;
  case MUL: return a * b;
  case MAX: return (a > b)? a : b;
  case MIN: return (a < b)? a : b;
  default: return a;
  }
}

static inline
void _ACC_reduction_init(int* var, _ACC_reduction_kind kind)
{
  switch(kind){
  case PLUS: *var = 0; return;
  case MUL: *var = 1; return;
  case MAX: *var = INT_MIN; return;
  case MIN: *var = INT_MAX; return;
  case BITAND: *var = ~0; return;
  case BITOR: *var = 0; return;
  case BITXOR: *var = 0; return;
  case LOGAND: *var = 1; return;
  case LOGOR: *var = 0; return;
  }
}

static inline
void _ACC_reduction_init(float* var, _ACC_reduction_kind kind)
{
  switch(kind){
  case PLUS: *var = 0.0f; return;
  case MUL: *var = 1.0f; return;
  case MAX: *var = FLT_MIN; return;
  case MIN: *var = FLT_MAX; return;
  default: return;
  }
}

static inline
void _ACC_reduction_init(double* var, _ACC_reduction_kind kind)
{
  switch(kind){
  case PLUS: *var = 0.0; return;
  case MUL: *var = 1.0; return;
  case MAX: *var = DBL_MIN; return;
  case MIN: *var = DBL_MAX; return;
  default: return;
  }
}


#define PZSIZE_T		long long
#define PZ_INNER_BLK		1


#define MAX_PERF_NUM 16
#define MAX_CITY_NUM 16



/////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////

/*
 * Pe 内の加算
 */
static //inline
float __ReductionPe( float v )
{
	
	const int stackSize = 2048;
	const int blk = stackSize / sizeof(float);

	volatile float tmp[ 1 ];
	
	const int tid = get_tid();
	const int max = get_maxtid();

	
	float sum;
	f_sw( &tmp[ 0 ], 0, v );

	wait_pe();

	if ( 0 == tid )
	{
		float v1,v2,v3,v4,v5,v6, v7;


		f_ld    ( &tmp[  0*blk ], 0x00, sum );
		f_ld    ( &tmp[ -1*blk ], 0x00, v1  );
//		chgthread();
		f_ld_add( &tmp[ -2*blk ], 0x00, v2, sum, v, v1  );
		f_ld_add( &tmp[ -3*blk ], 0x00, v3, sum, sum, v2  );
		f_ld_add( &tmp[ -4*blk ], 0x00, v4, sum, sum, v3  );
		f_ld_add( &tmp[ -5*blk ], 0x00, v5, sum, sum, v4  );
		f_ld_add( &tmp[ -6*blk ], 0x00, v6, sum, sum, v5  );
		f_ld_add( &tmp[ -7*blk ], 0x00, v7, sum, sum, v6  );
		sum += v7;
	}
	
	wait_pe();

	return sum;
}

/*
 * Pe 内の加算
 */
static //inline
double __ReductionPe( double v )
{
	
	const int stackSize = 2048;
	const int blk = stackSize / sizeof(double);
	
	volatile double tmp[1];
	const int tid = get_tid();
	const int max = get_maxtid();

	
	double sum;
	d_sd( &tmp[ 0 ], 0, v );
	wait_pe();

	if ( 0 == tid )
	{
		double v0,v1,v2,v3,v4,v5,v6,v7;
		double sum0, sum1;

		d_ld    ( &tmp[      0 ], 0x00, v0 );
		d_ld    ( &tmp[ -1*blk ], 0x00, v1  );
		d_ld    ( &tmp[ -2*blk ], 0x00, v2  );
//		chgthread();
		d_ld_add( &tmp[ -3*blk ], 0x00, v3, sum0, v0  , v1  );
		d_ld_add( &tmp[ -4*blk ], 0x00, v4, sum1, v2  , v3  );
		d_ld_add( &tmp[ -5*blk ], 0x00, v5, sum0, sum0, v4  );
		d_ld_add( &tmp[ -6*blk ], 0x00, v6, sum1, sum1, v5  );
		d_ld_add( &tmp[ -7*blk ], 0x00, v7, sum0, sum0, v6  );

		sum1 = sum1 + v7;
		sum  = sum0 + sum1;
	}

	
	wait_pe();

	return sum;
}

template< typename T >
static inline
T __ReductionCityForPeSingle( T v )
{
	const int tid    = get_tid();
	const int maxtid = get_maxtid();
	int pid = get_pid(); //  % MAX_CITY_NUM;
	int pidCity = pid / MAX_CITY_NUM * MAX_CITY_NUM;
	
	static T tmp[ 1024 ];
	
	// まずスレッド内でまとめる。
	v = __ReductionPe( v );
	
	// 1 個書く
	if ( 0 == tid )
	{
		tmp[ pid ] = v;
	}
	
	flush_L1();
	
	wait_city();
	
	int offset = pidCity;
	
	
	T sum = 0;
	if ( pid == pidCity )
	{
#ifdef DO_PREFETCH
	do_prefetch_L1( &tmp[ offset + tid          ], 0 );
#endif
		sum += tmp[ offset + tid          ];
		sum += tmp[ offset + tid + maxtid ];

		// もう一回まとめる
		sum = __ReductionPe( sum );
	}

	wait_city();
	
	return sum;
}


template< typename T >
static inline
T __ReductionPrefectureForPeSingle( T v )
{
	int tid     = get_tid();
	int maxtid  = get_maxtid();
	int pid     = get_pid(); //  % MAX_CITY_NUM;
	int cid     = pid / MAX_CITY_NUM;
	    pid     = pid % MAX_CITY_NUM;
	
	static T tmp[ 64 ];
	
	// まずCity内でまとめる。
	v = __ReductionCityForPeSingle( v );
	
	// 1 個書く
	if ( 0 == pid && tid == 0 )
	{
		tmp[ cid ] = v;
	}
	
	flush_L2();
	
	int offset = cid / 16 * 16;

	wait_prefecture();
#ifdef DO_PREFETCH
	do_prefetch_L1( &tmp[ offset          ], 0 );
#endif
	
	T sum = 0;

	if ( offset * MAX_CITY_NUM == get_pid() )
	{
		sum = tmp[ tid + offset          ];
		sum += tmp[ tid + offset + maxtid ];
		// もう一回まとめる
		sum = __ReductionPe( sum );
	}
	
	wait_prefecture();

	return sum;
}


template<typename T>
T __ReductionStateForPeSingle( T v )
{
	int tid     = get_tid();
	int maxtid  = get_maxtid();
	int pid     = get_pid();
	
	static T tmp[ 8 ];
	
	// まずPrefecture内でまとめる。
	v = __ReductionPrefectureForPeSingle( v );
	
	// 1 個書く
	#define PE_NUM  256
	#define PREF_NUM 4
	
	T sum = 0;

	sync();

	int prefid = pid / PE_NUM;
	if ( (0 == tid) && (0 == ( pid & (PE_NUM-1) ) ))
	{
		tmp[ prefid ] = v;
	}
	
	flush();
	
	if ( 0 == pid )
	{
#ifdef DO_PREFETCH
	  do_prefetch_L1( tmp, 0 );
#endif

#if 1
		T v = 0;
		if ( tid < 4 )
		{
			v = tmp[ tid ];
		}
		
		sum = __ReductionPe( v );
#else
		if(tid == 0){
		  sum = tmp[0];
		  sum += tmp[1];
		  sum += tmp[2];
		  sum += tmp[3];
		}
#endif
	}
	
	return sum;
}


///////////////////////////////////////////////////////////////////////////////


template<typename T>
static inline
void _ACC_gpu_init_reduction_var(T* v, int kind)
{
  _ACC_reduction_init(v, static_cast<_ACC_reduction_kind>(kind));
}

template<typename T>
static inline
void _ACC_gpu_init_reduction_var_single(T* v, int kind)
{
  if(get_tid() == 0){
    _ACC_reduction_init(v, static_cast<_ACC_reduction_kind>(kind));
  }
}

template<typename T>
static inline
void _ACC_pzcl_reduction_Th(T* r, int kind, T v)
{
  // 全スレッドで実行する
  const T tmp = __ReductionPe(v);

#if 0
  // なぜかこちらだとエラーが出る
  if(get_tid() == 0){
    *r = _ACC_reduction_op(*r, tmp, static_cast<_ACC_reduction_kind>(kind));
  }
#else
  *r = _ACC_reduction_op(*r, tmp, static_cast<_ACC_reduction_kind>(kind));
#endif
}

template<typename T>
static inline
void _ACC_pzcl_reduction_PrCiViPeTh(T* r, int kind, T v)
{
  const T tmp = __ReductionStateForPeSingle(v);
  if(get_pid() == 0 && get_tid() == 0){
    *r = _ACC_reduction_op(*r, tmp, static_cast<_ACC_reduction_kind>(kind));
  }
}



#endif //_ACC_PEZY_REDUCTION
