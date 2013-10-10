#include <nata/nata_rcsid.h>
__rcsId("$Id: nata_uid.cpp 86 2012-07-30 05:33:07Z m-hirano $")

#include <nata/libnata.h>



static union {
    uint64_t uRndBuf64[2];
    uint32_t uRndBuf32[4];
} uRndBuf = { { 0LL, 0LL } };

static uint64_t initSerial = 0LL;

#define rndBuf	uRndBuf.uRndBuf64
#define rnd0Ptr	&(uRndBuf.uRndBuf32[0])
#define rnd1Ptr	&(uRndBuf.uRndBuf32[1])
#define rnd2Ptr	&(uRndBuf.uRndBuf32[2])
#define rnd3Ptr	&(uRndBuf.uRndBuf32[3])

#define serialBuf	rndBuf[1]
#define incrSerial(x)	(x)++
#define needRefresh(x)	((x) == initSerial) ? true : false

static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

static char aTab[64] = {
    /*  0 */ 'A', /*  1 */ 'B', /*  2 */ 'C', /*  3 */ 'D',
    /*  4 */ 'E', /*  5 */ 'F', /*  6 */ 'G', /*  7 */ 'H',
    /*  8 */ 'I', /*  9 */ 'J', /* 10 */ 'K', /* 11 */ 'L',
    /* 12 */ 'M', /* 13 */ 'N', /* 14 */ 'O', /* 15 */ 'P',
    /* 16 */ 'Q', /* 17 */ 'R', /* 18 */ 'S', /* 19 */ 'T',
    /* 20 */ 'U', /* 21 */ 'V', /* 22 */ 'W', /* 23 */ 'X',
    /* 24 */ 'Y', /* 25 */ 'Z', /* 26 */ 'a', /* 27 */ 'b',
    /* 28 */ 'c', /* 29 */ 'd', /* 30 */ 'e', /* 31 */ 'f',
    /* 32 */ 'g', /* 33 */ 'h', /* 34 */ 'i', /* 35 */ 'j',
    /* 36 */ 'k', /* 37 */ 'l', /* 38 */ 'm', /* 39 */ 'n',
    /* 40 */ 'o', /* 41 */ 'p', /* 42 */ 'q', /* 43 */ 'r',
    /* 44 */ 's', /* 45 */ 't', /* 46 */ 'u', /* 47 */ 'v',
    /* 48 */ 'w', /* 49 */ 'x', /* 50 */ 'y', /* 51 */ 'z',
    /* 52 */ '0', /* 53 */ '1', /* 54 */ '2', /* 55 */ '3',
    /* 56 */ '4', /* 57 */ '5', /* 58 */ '6', /* 59 */ '7',
    /* 60 */ '8', /* 61 */ '9', /* 62 */ '@', /* 63 */ '_',
};



static void
base64Encode(const uint8_t *src, size_t len, uint8_t *dst) {
    size_t i;
    size_t lenM3 = len % 3;
    size_t modShort = (lenM3 == 0) ?
        0 : 3 - lenM3; /* How much we are short for complete 3
                        * bytes chunks. */
    size_t canoLen = len + modShort;
    size_t lastLoop = len - lenM3;
    int c;
    int e0, e1, e2, e3;
    int n = 0;

    for (i = 0; i < canoLen; i += 3) {
        /* Get 3 bytes */
        c = ((src[i] & 0xff) << 16);
        if (i < lastLoop || modShort == 0) {
            c |= (((src[i + 1] & 0xff) << 8) |
                  ((src[i + 2] & 0xff)));
        } else {
            if (modShort == 1) {
                c |= ((src[i + 1] & 0xff) << 8);
            }
        }

        /* Split the 3 bytes (24 bits) into 4 bytes (each 6 bits
         * chunk.) */
        e0 = (c >> 18) & 0x3f;
        e1 = (c >> 12) & 0x3f;
        e2 = (c >> 6) & 0x3f;
        e3 = c & 0x3f;

        dst[n++] = aTab[e0];
        dst[n++] = aTab[e1];

        if (i < lastLoop || modShort == 0) {
            dst[n++] = aTab[e2];
            dst[n++] = aTab[e3];
        } else {
            if (modShort == 1) {
                dst[n++] = aTab[e2];
            }
        }
    }
    dst[n] = '\0';
}


static void
refreshRnd(void) {
#if defined(NATA_API_POSIX)
    *rnd0Ptr = (uint32_t)random();
    *rnd1Ptr = (uint32_t)random();
    *rnd2Ptr = (uint32_t)random();
    *rnd3Ptr = (uint32_t)random();
#elif defined(NATA_API_WIN32API)
    *rnd0Ptr = (uint32_t)rand();
    *rnd1Ptr = (uint32_t)rand();
    *rnd2Ptr = (uint32_t)rand();
    *rnd3Ptr = (uint32_t)rand();
#else
#error Unknown/Non-supported API.
#endif /* NATA_API_POSIX, NATA_API_WIN32API */
    initSerial = serialBuf;
    incrSerial(serialBuf);
}



void
nata_initUid(void) {
    uint32_t seed;
#ifdef NATA_API_POSIX
    struct timespec ts;
#endif /* NATA_API_POSIX */

    pthread_mutex_lock(&lock);

#if defined(NATA_API_POSIX)
    errno = 0;
    if (clock_gettime(CLOCK_REALTIME, &ts) != 0) {
        perror("clock_gettime");
        seed = (uint32_t)time(NULL);
    } else {
        seed = (uint32_t)ts.tv_sec ^ (uint32_t)ts.tv_nsec;
    }
    srandom(seed);
#elif defined(NATA_API_WIN32API)
    {
        uint64_t l;
        QueryPerformanceCounter((LARGE_INTEGER *)&l);
        seed = (uint32_t)time(NULL) ^ (uint32_t)((l >> 32) ^ (l & 0xffffffff));
        srand(seed);
    }
#else
#error Unknown/Non-supported API.
#endif /* NATA_API_POSIX, NATA_API_WIN32API */
    refreshRnd();

    pthread_mutex_unlock(&lock);
}


void
nata_getUid(nata_Uid *uPtr) {
    uint64_t buf[2];
    nata_SHA1_Digest d;

    pthread_mutex_lock(&lock);    

    incrSerial(serialBuf);
    if (needRefresh(serialBuf) == true) {
        refreshRnd();
    }
    buf[0] = rndBuf[0];
    buf[1] = rndBuf[1];

    pthread_mutex_unlock(&lock);

    nata_getSha1((const uint8_t *)buf, sizeof(buf), d);
    base64Encode((const uint8_t *)d, sizeof(d), (uint8_t *)uPtr);
}
