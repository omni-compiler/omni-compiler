/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#ifndef _XMP_TLOG_H_
#define _XMP_TLOG_H_

#include "tlog_mpi.h"

#define _XMP_M_TLOG_TASK_IN(dummy)      tlog_log(TLOG_EVENT_1_IN)
#define _XMP_M_TLOG_TASK_OUT(dummy)     tlog_log(TLOG_EVENT_1_OUT)

#define _XMP_M_TLOG_LOOP_IN(dummy)      tlog_log(TLOG_EVENT_2_IN)
#define _XMP_M_TLOG_LOOP_OUT(dummy)     tlog_log(TLOG_EVENT_2_OUT)

#define _XMP_M_TLOG_REFLECT_IN(dummy)   tlog_log(TLOG_EVENT_3_IN)
#define _XMP_M_TLOG_REFLECT_OUT(dummy)  tlog_log(TLOG_EVENT_3_OUT)

#define _XMP_M_TLOG_BARRIER_IN(dummy)   tlog_log(TLOG_EVENT_4_IN)
#define _XMP_M_TLOG_BARRIER_OUT(dummy)  tlog_log(TLOG_EVENT_4_OUT)

#define _XMP_M_TLOG_REDUCTION_IN(dummy)         tlog_log(TLOG_EVENT_5_IN)
#define _XMP_M_TLOG_REDUCTION_OUT(dummy)        tlog_log(TLOG_EVENT_5_OUT)

#define _XMP_M_TLOG_BCAST_IN(dummy)     tlog_log(TLOG_EVENT_6_IN)
#define _XMP_M_TLOG_BCAST_OUT(dummy)    tlog_log(TLOG_EVENT_6_OUT)

#define _XMP_M_TLOG_GMOVE_IN(dummy)     tlog_log(TLOG_EVENT_7_IN)
#define _XMP_M_TLOG_GMOVE_OUT(dummy)    tlog_log(TLOG_EVENT_7_OUT)

extern void _XMP_tlog_init(void);
extern void _XMP_tlog_finalize(void);

#endif /* _XMP_TLOG_H_ */
