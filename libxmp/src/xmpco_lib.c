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

int _XMPCO_get_currentThisImage()
{
  return _XMP_get_execution_nodes()->comm_rank + 1;
}

MPI_Comm _XMPCO_get_currentComm()
{
  MPI_Comm *commp;
  commp = (MPI_Comm*)_XMP_get_execution_nodes()->comm;
  return *commp;
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


/*  get image2 of MPI communicatior comm2 corresponding to image1 of comm1.
 */
int _XMPCO_transImage_withComm(MPI_Comm comm1, int image1, MPI_Comm comm2)
{
  int image2, rank1, rank2;
  MPI_Group group1, group2;
  int stat1, stat2, stat3;

  rank1 = image1 - 1;
  stat1 = MPI_Comm_group(comm1, &group1);
  stat2 = MPI_Comm_group(comm2, &group2);

  stat3 = MPI_Group_translate_ranks(group1, 1, &rank1, group2, &rank2);
  //                           (in:Group1, n, rank1[n], Group2, out:rank2[n])
  if (rank2 == MPI_UNDEFINED)
    image2 = 0;
  else 
    image2 = rank2 + 1;

  if (stat1 != 0 || stat2 != 0 || stat3 != 0)
    _XMPCO_fatal("INTERNAL: _transimage_withComm failed with "
                       "stat1=%d, stat2=%d, stat3=%d",
                       stat1, stat2, stat3);

  _XMPCO_debugPrint("***IMAGE NUMBER translated from %d to %d\n",
                          image1, image2);

  return image2;
}


int _XMPCO_transImage_current2initial(int image)
{
  int num = _XMPCO_get_currentNumImages();

  if (image <= 0 || num < image)
    _XMPCO_fatal("ERROR: image index (%d) not specified within the range (1 to %d)\n",
                       image, num);

  if (!_XMPCO_is_subset_exec())
    return image;

  int initImage = _XMPCO_transImage_withComm(_XMPCO_get_currentComm(), image,
                                             MPI_COMM_WORLD);

  _XMPCO_debugPrint("*** got the initial image (%d) from the current image (%d)\n",
                    initImage, image);

  return initImage;
}


/*  get the initial image index corresponding to the image index
 *  of the nodes that the coarray is mapped to.
 */
int _XMPCO_get_initial_image_withDescPtr(int image, void *descPtr)
{
  if (descPtr == NULL)
    return _XMPCO_transImage_current2initial(image);

  MPI_Comm nodesComm =
    _XMPCO_get_comm_fromCoarrayInfo((CoarrayInfo_t*)descPtr);
  if (nodesComm == MPI_COMM_NULL)
    return _XMPCO_transImage_current2initial(image);

  // The coarray is specified with a COARRAY directive.

  int initImage =  _XMPCO_transImage_withComm(nodesComm, image,
                                             MPI_COMM_WORLD);

  _XMPCO_debugPrint("*** got the initial image (%d) from the image mapping to nodes (%d)\n",
                    initImage, image);

  return initImage;
}

