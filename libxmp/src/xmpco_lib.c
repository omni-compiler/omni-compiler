#include <string.h>
#include "xmpco_internal.h"

static int _initialThisImage;
static int _initialNumImages;

/*****************************************\
  initial images
\*****************************************/

void _XMPCO_set_initialThisImage()
{
  int rank;

  if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != 0)
    _XMPCO_fatal("INTERNAL ERROR: "
                 "MPI_Comm_rank(MPI_COMM_WORLD, ) failed");

  _initialThisImage = rank + 1;
}

void _XMPCO_set_initialNumImages()
{
  int size;

  if (MPI_Comm_size(MPI_COMM_WORLD, &size) != 0)
    _XMPCO_fatal("INTERNAL ERROR: "
                 "MPI_Comm_size(MPI_COMM_WORLD, ) failed");
  _initialNumImages = size;
}

int _XMPCO_get_initialThisImage()
{
  return _initialThisImage;
}

int _XMPCO_get_initialNumImages()
{
  return _initialNumImages;
}


/*****************************************\
  current images
\*****************************************/

int _XMPCO_get_currentNumImages()
{
  return _XMP_get_execution_nodes()->comm_size;
}

/*********
int _XMPCO_get_currentMPIComm()
{
  return *(_XMP_get_execution_nodes()->comm);
}
**********/

int _XMPCO_get_currentThisImage()
{
  return _XMP_get_execution_nodes()->comm_rank + 1;
}


BOOL _XMPCO_is_subset_exec()
{
  if (_XMPCO_get_currentNumImages() < _initialNumImages)
    // now executing in a task region
    return TRUE;
  if (_XMPF_coarray_get_image_nodes() != NULL)
    // image directive is now valid
    return TRUE;
  return FALSE;
}


