static char rcsid[] = "";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "omni.h"


extern char	*optarg;
extern int	optind;


#if defined(__OMNI_SCASH__) || defined(__OMNI_SHMEM__)
#pragma omp threadprivate(optind,optarg)
#endif


usage (name)
     char *name;
{
  char *cmd;

  cmd = strrchr (name, '/');
  if (cmd == NULL) {
    cmd = name;
  } else {
    cmd += 1;
  }

  printf ("%s : [options]\n", cmd);
  printf ("--------------\n");
  printf ("-i : set thread number, exit function called by this thread.\n");
  printf ("-c : set exit code\n");
  printf ("-h : this message.\n");

  exit (0);
}


int
main (argc, argv)
     int	argc;
     char	*argv[];
{

  int	eid = 0;
  int	ec  = 0;
  int	c;

  while ((c = getopt (argc, argv, "i:c:h")) != EOF) {
    switch (c) {
    case 'i':
      eid = atoi(optarg);
      break;
    case 'c':
      ec = atoi(optarg);
      break;
    case 'h':
    default:
      usage(argv[0]);
      break;
    }
  }

  #pragma omp parallel 
  {
    if (omp_get_thread_num () == eid) {
      exit (ec);
    }
  }

  return 0;
}
