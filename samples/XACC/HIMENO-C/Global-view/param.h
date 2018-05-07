#if defined(SIZE_XS)
//XSMALL
#define MX0     33
#define MY0     33
#define MZ0     65
#define MIMAX     32
#define MJMAX     32
#define MKMAX     64

#elif defined(SIZE_S)
//SMALL
#define MX0     65
#define MY0     65
#define MZ0     129
#define MIMAX     64
#define MJMAX     64
#define MKMAX     128

#elif defined(SIZE_M)
//MIDDLE
#define MX0     129
#define MY0     129
#define MZ0     257
#define MIMAX     128
#define MJMAX     128
#define MKMAX     256

#elif defined(SIZE_L)
//LARGE
#define MX0     257
#define MY0     257
#define MZ0     513
#define MIMAX     256
#define MJMAX     256
#define MKMAX     512

#elif defined(SIZE_XL)
//XLARGE
#define MX0     513
#define MY0     513
#define MZ0     1025
#define MIMAX     512
#define MJMAX     512
#define MKMAX     1024

#endif
