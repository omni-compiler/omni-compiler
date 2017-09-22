#ifndef _XMPCO_PARAMS_H
#define _XMPCO_PARAMS_H

#define BOOL   int
#define TRUE   1
#define FALSE  0


/** Threshold of memory size to share in the pool
 */
#define _XMPCO_default_poolThreshold     (40*1024*1024)      // 40MB

/** Size of the communication buffer prepared for short communications
 *  to avoid allocation and registration every communication time
 */
#define _XMPCO_default_localBufSize      (400000)            // ~400kB

#define _XMPCO_default_isMsgMode         FALSE
#define _XMPCO_default_isSafeBufferMode  FALSE
#define _XMPCO_default_isSyncPutMode     FALSE
#define _XMPCO_default_isEagerCommMode   FALSE


/** COMM_UNIT   : minimum unit of size for PUT/GET communication
 *  MALLOC_UNIT : minimum unit of size for memory allocation
 *                MALLOC_UNIT must be divisible by COMM_UNIT
 */
#if defined(_XMP_FJRDMA)
# define COMM_UNIT      ((size_t)4)
# define MALLOC_UNIT    ((size_t)8)
# define ONESIDED_COMM_LAYER "FJRDMA"
#elif defined(_XMP_GASNET)
# define COMM_UNIT      ((size_t)1)
# define MALLOC_UNIT    ((size_t)4)
# define ONESIDED_COMM_LAYER "GASNET"
#elif defined(_XMP_MPI3_ONESIDED)
# define COMM_UNIT      ((size_t)1)
# define MALLOC_UNIT    ((size_t)4)
# define ONESIDED_COMM_LAYER "MPI3"
#else
# define COMM_UNIT      ((size_t)1)
# define MALLOC_UNIT    ((size_t)4)
# define ONESIDED_COMM_LAYER "unknown"
#endif


/*****************************************\
  set functions
\*****************************************/
extern void _XMPCO_set_poolThreshold(unsigned size);
extern void _XMPCO_set_localBufSize(unsigned size);
extern void _XMPCO_set_isMsgMode(BOOL sw);
extern void _XMPCO_set_isMsgMode_quietly(BOOL sw);
extern void _XMPCO_set_isSafeBufferMode(BOOL sw);
extern void _XMPCO_set_isSyncPutMode(BOOL sw);
extern void _XMPCO_set_isEagerCommMode(BOOL sw);

extern void _XMPCO_reset_isMsgMode(void);

/*****************************************\
  get functions
\*****************************************/
extern unsigned _XMPCO_get_poolThreshold(void);
extern size_t   _XMPCO_get_localBufSize(void);
extern BOOL     _XMPCO_get_isMsgMode(void);
extern BOOL     _XMPCO_get_isSafeBufferMode(void);
extern BOOL     _XMPCO_get_isSyncPutMode(void);
extern BOOL     _XMPCO_get_isEagerCommMode(void);


#endif /*_XMPCO_PARAMS_H*/
