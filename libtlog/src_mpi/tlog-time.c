static char rcsid[] = "$Id: tlog-time.c,v 1.1.1.1 2005/06/20 09:56:18 msato Exp $";
/* 
 * $Release$
 * $Copyright$
 */
/*
 *  TIMER routine
 *
 *  $Id: tlog-time.c,v 1.1.1.1 2005/06/20 09:56:18 msato Exp $
 */
#include <stddef.h>

/* 
 * using gettimeofday as default
 */

#if defined(__TTIME_MPI_WTIME__)
/*
 *  TIMER routine for MPI
 */
double
tlog_timestamp()
{
    return MPI_Wtime();
}

void tlog_timestamp_init(){ }

/**  BEGINNING OF MACHINE-DEPENDENT PART  **/
/***  SUN SOLARIS  ***/
#elif defined(OMNI_OS_SOLARIS)

#include <sys/time.h>

#ifdef __GNUC__
#define USE_LL
#endif

#if __STDC__ - 0 == 0 && !defined(_NO_LONGLONG)
#define USE_LL
#endif

double
tlog_timestamp()
{
#ifdef USE_LL
    return ((double)gethrtime()) * .000000001;
#else
    hrtime_t hrT = gethrtime();
    return hrT._d * .000000001;
#endif /* USE_LL */
}

void tlog_timestamp_init(){ }

/***  SGI IRIX  ***/
#elif defined(OMNI_OS_IRIX)
#include <time.h>

double
tlog_timestamp()
{
    struct timespec t1;
    clock_gettime(CLOCK_SGI_CYCLE, &t1);
    return (double)t1.tv_sec + (double)t1.tv_nsec * .000000001;
}

void tlog_timestamp_init(){ }

/**  END OF MACHINE-DEPENDENT PART  **/
#else /* defaults */

/***  gettimeofday(2)  ***/
#include <sys/time.h>

static struct timeval t0;

void tlog_timestamp_init()
{
    gettimeofday(&t0, NULL);
}

double
tlog_timestamp()
{
    struct timeval t1;
    gettimeofday(&t1, NULL);

    return (double)(t1.tv_sec-t0.tv_sec) + (double)(t1.tv_usec-t0.tv_usec) * .000001;
}

#endif
