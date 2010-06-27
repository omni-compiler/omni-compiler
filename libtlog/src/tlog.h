/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
#ifndef _TLOG_H
#define _TLOG_H

#define MAX_THREADS 64
#define TLOG_FILE_NAME "t.log"

#define TLOG_BLOCK_SIZE 1024
typedef struct _TLOG_BLOCK {
    struct _TLOG_BLOCK *next;
    double data[TLOG_BLOCK_SIZE/sizeof(double)];
} TLOG_BLOCK;

typedef enum tlog_type {
    TLOG_UNDEF = 0,	/* undefined */
    TLOG_END = 1, 	/* END*/
    TLOG_START = 2,
    TLOG_RAW = 3, 	/* RAW information */
    TLOG_EVENT = 4,
    TLOG_EVENT_IN = 5,
    TLOG_EVENT_OUT = 6,
    TLOG_FUNC_IN = 7,
    TLOG_FUNC_OUT = 8,
    TLOG_BARRIER_IN = 9,
    TLOG_BARRIER_OUT = 10,
    TLOG_PARALLEL_IN = 11,
    TLOG_PARALLEL_OUT = 12,
    TLOG_CRITICAL_IN = 13,
    TLOG_CRITICAL_OUT = 14,
    TLOG_LOOP_INIT_EVENT = 15,
    TLOG_LOOP_NEXT_EVENT = 16,
    TLOG_SECTION_EVENT = 17,
    TLOG_SIGNLE_EVENT = 18,
    TLOG_END_END
} TLOG_TYPE;

/* every log record is 2 double words. */
typedef struct tlog_record {
    char log_type;	/* major type */
    char proc_id;	/* processor id */
    _omInt16_t arg1;	/* minor type */
    _omInt32_t arg2;
    double time_stamp;
} TLOG_DATA;

typedef struct tlog_handle {
    TLOG_BLOCK *block_top;
    TLOG_BLOCK *block_tail;
    TLOG_DATA *free_p;
    TLOG_DATA *end_p;
} TLOG_HANDLE;
 
extern TLOG_HANDLE tlog_handle_table[];

/* prototypes */
void tlog_init(char *name);
void tlog_finalize(void);
void tlog_log(int id,enum tlog_type type);
void tlog_log1(int id,TLOG_TYPE type,int arg1);
void tlog_log2(int id,TLOG_TYPE type,int arg1,int arg2);
double tlog_timestamp(void);

#endif /* _TLOG_H */
