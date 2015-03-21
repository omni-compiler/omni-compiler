#include "xmpf_internal.h"

static void _coarray_msg(int sw);

#define DIV_CEILING(m,n)  (((m)-1)/(n)+1)


/*****************************************\
  runtime environment
\*****************************************/

int _XMPF_coarrayMsg = 0;          // default: message off
int _XMPF_coarrayErr = 0;          // default: aggressive error check off

void _XMPF_coarray_init(void)
{
  char *str;

  if (xmp_node_num() == 1) {
    str = getenv("XMPF_COARRAY_MSG1");
    if (str != NULL) {
      _coarray_msg(atoi(str));
      return;
    }
  }

  str = getenv("XMPF_COARRAY_MSG");
  if (str != NULL) {
    _coarray_msg(atoi(str));
  }
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

  _XMPF_coarrayDebugPrint("xmpf_coarray_msg ON\n"
                          "  %zd-byte boundary, using %s\n",
                          BOUNDARY_BYTE,
#if defined(_XMP_COARRAY_FJRDMA)
                          "FJRDMA"
#elif defined(_XMP_COARRAY_GASNET)
                          "GASNET"
#else
                          "something unknown"
#endif
                          );
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
    _XMPF_coarrayFatal("current rextriction: cannot use %s in any task construct",
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
  fprintf(stderr, "CAF[%d] %s", xmp_node_num(), work);
  va_end(list);
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


