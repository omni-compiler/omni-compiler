#include "xmpf_internal.h"

static void _coarray_msg(int sw);

#define DIV_CEILING(m,n)  (((m)-1)/(n)+1)

/*****************************************\
  runtime environment
\*****************************************/

int _XMPF_coarrayMsg = 0;          // default: message off
int _XMPF_coarrayErr = 0;          // default: aggressive error check off

/*
 * read environment variable XMPF_COARRAY_MSG
 * usage: <v1><d><v2><d>...<vn>
 *    <vk>  value for image index k
 *    <d>   delimiter ',' or ' '
 */
void _XMPF_coarray_init(void)
{
  char *tok, *string;
  int i;
  char delim[] = ", ";

  string = strdup(getenv("XMPF_COARRAY_MSG"));
  tok = strtok(string, delim);
  for (i = 1; tok != NULL; i++, tok = strtok(NULL, delim)) {
    if (this_image_() == i)
      _coarray_msg(atoi(tok));
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

  _XMPF_coarrayDebugPrint("XMPF_COARRAY_MSG=%d\n"
                          "  %zd-byte boundary\n"
                          "  with %s\n",
                          sw, BOUNDARY_BYTE,
#if defined(_XMP_COARRAY_FJRDMA)
                          "FJRDMA"
#elif defined(_XMP_COARRAY_GASNET)
                          "GASNET"
#else
                          "(something unknown)"
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
  fprintf(stderr, "CAF[%d] %s", this_image_(), work);
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


