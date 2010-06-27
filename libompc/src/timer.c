/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 *
 * @file timer.c
 */

/* timer functions */
#include <stdio.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>

/* #define USE_GETRUSAGE */

double second()
{
    double t;
#ifdef USE_GETRUSAGE
    struct rusage ru;
 
    getrusage(RUSAGE_SELF, &ru);
    t = (double)(ru.ru_utime.tv_sec+ru.ru_stime.tv_sec) + 
      ((double)(ru.ru_utime.tv_usec+ru.ru_stime.tv_usec))/1.0e6 ;
    return t ;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    t = (double)(tv.tv_sec) + ((double)(tv.tv_usec))/1.0e6;
    return t ;
#endif
}

static double start_time;

double start_timer()
{
    start_time = second();
    return start_time;
}

double lap_time(char *msg)
{
    double t;
    t = second();
    if (msg)
      printf("%s %f\n", msg, t-start_time);
    return t;
}

