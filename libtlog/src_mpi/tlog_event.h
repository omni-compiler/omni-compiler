#ifndef _TLOG_EVENT_H
#define _TLOG_EVENT_H

#define TLOG_PROF_N_IDX 512
#define TLOG_PROF_SHIFT 9
#define TLOG_PROF_IDX(n) ((n) >> TLOG_PROF_SHIFT)
#define TLOG_PROF_OFF(n) ((n) & (TLOG_PROF_N_IDX-1))

typedef TLOG_DATA tlog_exec_event;

typedef struct _exec_events {
  int n_data;
  tlog_exec_event *data[TLOG_PROF_N_IDX];
} tlog_exec_events;

typedef struct _exec_profile {
  int n_node;
  tlog_exec_events node_data[1]; /* size must be n_node */
} tlog_exec_profile;

tlog_exec_profile *tlog_read_file(char *fname,int n_node);
int tlog_get_N_events(tlog_exec_profile *ep, int node);
tlog_exec_event *tlog_get_event(tlog_exec_profile *ep,int node, int i);

#endif /* _TLOG_EVENT_H */




