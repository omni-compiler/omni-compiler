#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include "tlog_mpi.h"
#include "tlog_event.h"

union {
  char u8[8];
  double d;
} u8_d;

tlog_exec_profile *tlog_read_file(char *fname,int n_node)
{
  TLOG_DATA d;
  FILE *fp;
  int i,idx;
  tlog_exec_profile *ep;
  tlog_exec_events *epp;
  tlog_exec_event *dp;
  int is_bigendian;

  //char t;
  union {
    long i;
    char c;
  } x;

  x.i = 1;
  is_bigendian = (x.c == 0);

  fp = fopen(fname,"r");
  if(fp == NULL){
    printf("cannot open %s\n",fname);
    exit(1);
  }
  ep = (tlog_exec_profile *)
    malloc(sizeof(tlog_exec_profile) + 
	   sizeof(tlog_exec_events)*(n_node-1));
  if(ep == NULL){
    fprintf(stderr,"tlog event profile cannot alloc");
    exit(1);
  }
  for(i = 0; i < n_node; i++)
    bzero(&ep->node_data[i],sizeof(tlog_exec_events));

  while(fread(&d,sizeof(d),1,fp) > 0){
    if(d.log_type == TLOG_UNDEF) continue;  /* skip */

    if(!is_bigendian) _tlog_block_swap_bytes(&d);

    if(d.proc_id >= n_node){
      fprintf(stderr,"exec profile: bad node, %d\n",d.proc_id);
      exit(1);
    }

    epp = &ep->node_data[d.proc_id];
    i = epp->n_data++;
    idx = TLOG_PROF_IDX(i);
    if((dp = epp->data[idx]) == NULL){
      dp = (tlog_exec_event *)
	malloc(sizeof(tlog_exec_event)*TLOG_PROF_N_IDX);
      if(dp == NULL){
	fprintf(stderr,"tlog exec profile, data, out of memory");
	exit(1);
      }
      epp->data[idx] = dp;
    }
    dp += TLOG_PROF_OFF(i);
    *dp = d;
  }
  fclose(fp);
  ep->n_node = n_node;
  return ep;
}

int tlog_get_N_events(tlog_exec_profile *ep, int node)
{
  if(node >= ep->n_node){
    fprintf(stderr,"get_exec_N_events, bad node, %d > %d\n",
	    node,ep->n_node);
    exit(1);
  }
  return ep->node_data[node].n_data;
}

tlog_exec_event *tlog_get_event(tlog_exec_profile *ep,int node, int i)
{
  if(node >= ep->n_node){
    fprintf(stderr,"get_exec_event, bad node, %d > %d\n",
	    node,ep->n_node);
    exit(1);
  }
  if(i >= ep->node_data[node].n_data){
    fprintf(stderr,"get_exec_evens, bad data idx, %d > %d\n",
	    i,ep->node_data[node].n_data);
    exit(1);
  }
  return ep->node_data[node].data[TLOG_PROF_IDX(i)]+TLOG_PROF_OFF(i);
}

/* for debug */
void tlog_exec_profile_dump(tlog_exec_profile *epp, int node)
{
  int i,n;
  tlog_exec_event *ep;

  n = epp->node_data[node].n_data;
  printf("Tlog exec profile dump: %d, #data=%d\n",node,n);
  for(i = 0; i < n; i++){
    ep = epp->node_data[node].data[TLOG_PROF_IDX(i)]+TLOG_PROF_OFF(i);
    printf("%d:proc_id=%d, log_type=%d, time=%g\n",
	   i,ep->proc_id, ep->log_type,ep->time_stamp);
  }
}




