static char rcsid[] = "$Id$";
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


void
check_thread_num ()
{
  printf ("number of threads is %d\n", omp_get_max_threads ());
}


void
check_dynamic ()
{
  if (omp_get_dynamic () == 0) {
    printf ("dynamic schedule is disable\n");
  } else {
    printf ("dynamic schedule is enable\n");
  }
}


void
check_nested ()
{
  if (omp_get_nested () == 0) {
    printf ("nested parallelism is disable\n");
  } else {
    printf ("nested parallelism is enable\n");
  }
}


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
  printf ("-t : print number of threads\n");
  printf ("-d : print dynamic schedule is enable/disable\n");
  printf ("-n : print nested parallelism is enable/disable\n");
  printf ("-h : this message.\n");

  exit (0);
}


main (argc, argv)
     int	argc;
     char	*argv[];
{
  extern char	*optarg;
  extern int	optind;

  int	c;

  while ((c = getopt (argc, argv, "tdnh")) != EOF) {
    switch (c) {
    case 't':
      check_thread_num ();
      break;
    case 'd':
      check_dynamic ();
      break;
    case 'n':
      check_nested ();
      break;
    case 'h':
    default:
      usage(argv[0]);
      break;
    }
  }

  return 0;
}
