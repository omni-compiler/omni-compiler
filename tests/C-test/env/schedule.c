static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
#include <stdlib.h>
#include <strings.h>

#include <omp.h>
#include "omni.h"


int	tprvt;
#pragma omp threadprivate (tprvt)


void
test_static (thds, stride_org)
     int	thds;
     int	stride_org;
{
  int	id, i, j, iter, stride;
  int	*buff;
  int	errors = 0;
  int	finish = 0;


  if (stride_org == 0) {
    stride = 10;
    iter = 10 * thds;
  } else {
    stride = stride_org;
    iter = stride * thds * 10;
  }

  buff = (int *) malloc (sizeof(int) * iter);
  if (buff == NULL) {
    printf ("can not allocate memory\n");
    exit (1);
  }

  #pragma omp parallel for schedule (runtime)
  for (i=0; i<iter; i++) {
    int	id = omp_get_thread_num ();
    buff[i] = id;

    if (id == 0) {
      while (finish < thds - 1) {
#ifdef __OMNI_SCASH__
	waittime (1);
#endif
	#pragma omp flush
      }
    } else {
      if (i == (iter - ((thds - id - 1) * stride) - 1)) {
	#pragma omp critical
	{
	  finish += 1;
	}
      }
    }
  }

  id = 0;
  for (i=0; i<iter; i+=stride) {
    for (j=0; j<stride; j++) {
      if (buff[i+j] != id) {
	errors += 1;
      }
    }
    id = (id + 1) % thds;
  }

  if (errors != 0) {
    printf ("scheduling test is FAILED(static,%d).\n",stride_org);
    exit (0);
  } else {
    printf ("scheduling test is SUCCESS(static,%d).\n",stride_org);
    exit (1);
  }
}


void
test_dynamic (thds, stride_org)
     int	thds;
     int	stride_org;
{
  int	id, i, j, iter, stride;
  int	*buff;
  int	errors = 0;
  int	finish = 0;

  if (stride_org == 0) {
    stride = 1;
    iter = 100 * thds;
  } else {
    stride = stride_org;
    iter = stride * thds * 100;
  }

  buff = (int *) malloc (sizeof(int) * iter);
  if (buff == NULL) {
    printf ("can not allocate memory.\n");
    exit (1);
  }


  #pragma omp parallel for schedule(runtime)
  for (i=0; i<iter; i++) {
    int id = omp_get_thread_num ();

    buff[i] = id;

    if (id == 0) {
      while (finish == 0) {
#ifdef __OMNI_SCASH__
	waittime (1);
#endif
	#pragma omp flush
      }
    } else {
      if (i == iter-1) {
	finish = 1;
        #pragma omp flush
      }
    }
  }

  for (i=0; i<iter; i+=stride) {
    id = buff[i];
    for (j=1; j<stride; j++) {
      if (id != buff[i+j]) {
	errors += 1;
      }
    }
  }

  for (i=0,j=0; i<iter; i++) {
    if (buff[i] == 0) {
      j++;
    }
  }
  if (j != stride && j != 0) {
    errors += 1;
  }


  tprvt = 0;
  #pragma omp parallel for schedule(runtime) copyin(tprvt)
  for (i=0; i<iter; i++) {
    int id = omp_get_thread_num ();

    buff[i] = id;

    if (tprvt == 0) {
      barrier (thds);
      tprvt = 1;
    }
  }

  for (i=0; i<iter; i+=stride) {
    id = buff[i];
    for (j=1; j<stride; j++) {
      if (id != buff[i+j]) {
	errors += 1;
      }
    }
  }


  if (errors != 0) {
    printf ("scheduling test is FAILED(dynamic,%d).\n",stride_org);
    exit (0);
  } else {
    printf ("scheduling test is SUCCESS(dynamic,%d).\n",stride_org);
    exit (1);
  }
}


void
test_guided (thds,stride_org)
     int	thds;
     int	stride_org;
{
  int	id, i, j, iter, stride, max_chunk;
  int	*buff;
  int	errors = 0;
  int	finish = 0;

  if (stride_org == 0) {
    stride = 1;
    iter = 100 * thds;
  } else {
    stride = stride_org;
    iter = stride * thds * 100;
  }

  buff = (int *) malloc (sizeof(int) * (iter + 1));
  if (buff == NULL) {
    printf ("can not allocate memory.\n");
    exit (1);
  }
  buff[iter] = -1;


  #pragma omp parallel for schedule(runtime)
  for (i=0; i<iter; i++) {
    int id = omp_get_thread_num ();

    buff[i] = id;

    if (id == 0) {
      while (finish == 0) {
#ifdef __OMNI_SCASH__
	waittime (1);
#endif
	#pragma omp flush
      }
    } else {
      if (i == iter-1) {
	finish = 1;
        #pragma omp flush
      }
    }
  }

  max_chunk = 0;
  for (i=0; i<iter; i+=j) {
    id = buff[i];
    for (j=1; buff[i+j]==id; j++) {
    }
    if (buff[i+j] != -1  &&  j < stride) {
      errors += 1;
    }
    max_chunk = (max_chunk<j)?(j):(max_chunk);
  }
  if (max_chunk <= stride) {
    errors += 1;
  }


  tprvt = 0;
  #pragma omp parallel for schedule(runtime) copyin(tprvt)
  for (i=0; i<iter; i++) {
    int id = omp_get_thread_num ();

    buff[i] = id;

    if (tprvt == 0) {
      tprvt = 1;
      barrier (thds);
    }

    if (i%stride == 0) {
      if ((iter*96)/100 < i/stride) {
	sleep(1);
      }
    }
  }

  max_chunk = 0;
  for (i=0; i<iter; i+=j) {
    id = buff[i];
    for (j=1; buff[i+j]==id; j++) {
    }
    if (buff[i+j] != -1  &&  j < stride) {
      errors += 1;
    }
#if 0
    if (j == stride) {
      cnt ++;
    }
#endif
    max_chunk = (max_chunk<j)?(j):(max_chunk);
  }
  if (max_chunk <= stride) {
    errors += 1;
  }
#if 0
  if (cnt == 0) {
    errors += 1;
  }
#endif


  if (errors != 0) {
    printf ("scheduling test is FAILED(guided,%d).\n",stride_org);
    exit (0);
  } else {
    printf ("scheduling test is SUCCESS(guided,%d).\n",stride_org);
    exit (1);
  }
}


void
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

  printf ("%s : SCHEDULE [STRIDE]\n", cmd);
  printf ("--------------\n");
  printf ("SCHEDULE : one of static, dynamic, guided\n");
  printf ("STRIDE   : set a number lager than 0\n");
  exit (1);
}


main (argc, argv)
     int	argc;
     char	*argv[];
{
  int	thds, stride = 0;


  thds = omp_get_max_threads ();
  if (thds == 1) { 
    printf ("should be run this program on multi threads.\n");
    exit (1);
  }
  omp_set_dynamic (0);


  if (!(argc == 3  ||  argc == 2)) {
    usage (argv[0]);
  }

  if (argc == 3) {
    stride = atoi(argv[2]);
    if (stride <= 0) {	
      usage (argv[0]);
    }
  }

  if (!strcmp ("static", argv[1])) {
    test_static (thds, stride);
  } else if (!strcmp ("dynamic", argv[1])) {
    test_dynamic (thds, stride);
  } else if (!strcmp ("guided", argv[1])) {
    test_guided (thds, stride);
  } else {
    usage (argv[0]);
  }

  return 0;
}
