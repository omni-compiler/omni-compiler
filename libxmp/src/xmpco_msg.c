#include "xmpco_internal.h"


void _XMPCO_fatal(char *format, ...)
{
  int rank;
  char work[800];

  va_list list;
  va_start(list, format);
  vsprintf(work, format, list);
  va_end(list);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  fprintf(stderr, "XMPCO [rank=%d] %s\n", rank, work);

  _XMP_fatal("...fatal error in XMP/F Coarray runtime");
}


void _XMPCO_debugPrint(char *format, ...)
{
  int current, initial;
  char work[800];

  if (! _XMPCO_get_isMsgMode())
    return;

  va_list list;
  va_start(list, format);
  vsprintf(work, format, list);
  va_end(list);

  current = _XMPCO_get_currentThisImage();
  initial = _XMPCO_get_initialThisImage();
  if (current == initial)
    fprintf(stderr, "XMPCO[%d] %s", initial, work);
  else
    fprintf(stderr, "XMPCO[%d(current=%d)] %s", initial, current, work);
}


