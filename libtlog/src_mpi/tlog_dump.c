#include <stdlib.h>
#include <stdio.h>
#include "tlog_mpi.h"
#include "tlog_event.h"

main(int argc, char *argv[])
{
  char *fname;
  tlog_exec_profile *epp;
  int n,n_node;

  if(argc != 3){
    printf("bad arg\n %d log_file_name n_node\n",argv[0]);
    exit(1);
  }
  fname = argv[1];
  n_node = atoi(argv[2]);
  if(n_node <= 0){
    printf("bad n_node %d\n",n_node);
    exit(1);
  }

  epp = tlog_read_file(fname,n_node);
  
  for(n = 0; n < epp->n_node; n++){
    printf("-------\n");
    tlog_exec_profile_dump(epp,n);
  }
}

