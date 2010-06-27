/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 *
 * timer.h
 */
/* timer definition */

#ifndef _TIMER_H
#define _TIMER_H

#ifdef __cplusplus
extern "C" {
#endif

double second(void);
double start_timer(void);
double lap_time(char *);

#ifdef __cplusplus
};
#endif

#endif /* _TIMER_H */
