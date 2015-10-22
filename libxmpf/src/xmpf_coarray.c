#include "xmpf_internal.h"

static void _coarray_msg(int sw);

#define DIV_CEILING(m,n)  (((m)-1)/(n)+1)

/*****************************************\
  runtime environment
\*****************************************/

int _XMPF_coarrayMsg = 0;          // default: message off
int _XMPF_coarrayErr = 0;          // default: aggressive error check off

/* initialization called in xmpf_main
 *  1. set static variable _this_image and _num_nodes
 *  2. read environment variable XMPF_COARRAY_MSG
 *     usage: <v1><d><v2><d>...<vn>
 *        <vk>  value for image index k
 *        <d>   delimiter ',' or ' '
 */
void _XMPF_coarray_init(void)
{
  /*
   *  set who-am-i
   */
  _XMPF_set_this_image();

  /*
   * read environment variable
   */
  char *tok, *work, *env;
  int i;
  char delim[] = ", ";

  env = getenv("XMPF_COARRAY_MSG");
  if (env == NULL) 
    return;

  work = strdup(env);
  tok = strtok(work, delim);
  for (i = 1; tok != NULL; i++, tok = strtok(NULL, delim)) {
    if (XMPF_this_image == i)
      _coarray_msg(atoi(tok));
  }
}


/* NOT USED
 */
void _XMPF_coarray_finalize(void)
{
  xmpf_sync_all_auto_();
}


/*
 *  hidden subroutine interface,
 *   which can be used in the user program
 */
void xmpf_coarray_msg_(int *sw)
{
  _coarray_msg(*sw);
}

void _coarray_msg(int sw)
{
  switch (sw) {
  case 0:
  default:
    _XMPF_coarrayDebugPrint("xmpf_coarray_msg OFF\n");
    _XMPF_coarrayMsg = 0;
    return;

  case 1:
    _XMPF_coarrayMsg = 1;
    break;
  }

  _XMPF_coarrayDebugPrint("XMPF_COARRAY_MSG=%d\n"
                          "  %u-byte boundary\n"
                          "  over %s\n",
                          sw, ONESIDED_BOUNDARY, COMM_LAYER);
}


/*****************************************\
  internal information
\*****************************************/


/*****************************************\
  restriction checker
\*****************************************/

int _XMPF_nowInTask()
{
  return xmp_num_nodes() < xmp_all_num_nodes();
}

void _XMPF_checkIfInTask(char *msgopt)
{
  if (_XMPF_nowInTask())
    _XMPF_coarrayFatal("current restriction: cannot use %s in any task construct",
                       msgopt);
}


void _XMPF_coarrayDebugPrint(char *format, ...)
{
  if (!_XMPF_coarrayMsg)
    return;

  char work[1000];
  va_list list;
  va_start(list, format);
  vsprintf(work, format, list);
  fprintf(stderr, "CAF[%d] %s", XMPF_this_image, work);
  va_end(list);
}

void xmpf_coarray_fatal_(char *msg, int *msglen)
{
  _XMPF_coarrayFatal("%*s", *msglen, msg);
}

void _XMPF_coarrayFatal(char *format, ...)
{
  char work[1000];
  va_list list;
  va_start(list, format);
  vsprintf(work, format, list);
  va_end(list);
  _XMP_fatal(work);
}


