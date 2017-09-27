#include <string.h>
#include "xmpco_internal.h"
#include "_xmpco_alloc.h"

static int _initialThisImage;
static int _initialNumImages;
// The current `this image' and `num images' are managed
// in the lower runtime library.

static _XMP_nodes_t *_imageDirNodes;


/***********************************************\
  Inquire functions -- THIS IMAGE  
\***********************************************/

void XMPCO_this_image_coarray(CoarrayInfo_t *cinfo, int corank, int image[])
{
  int size, index, image_coarray, magic;
  _XMP_nodes_t *nodes;

  nodes = cinfo->nodes;
  if (nodes != NULL) {
    image_coarray = _XMPCO_this_image_onNodes(nodes);
  } else {
    image_coarray = _XMPCO_get_currentThisImage();
  }

  if (image_coarray == 0) {    // This image is out of the nodes.
    for (int i = 0; i < corank; i++)
      image[i] = 0;
    return;
  }

  magic = image_coarray - 1;
  for (int i = 0; i < corank; i++) {
    size = cinfo->cosize[i];
    index = magic % size;
    image[i] = index + cinfo->lcobound[i];
    magic /= size;
  }
}


int XMPCO_this_image_coarray_dim(CoarrayInfo_t *cinfo, int corank, int dim)
{
  int size, index, image_coarray, magic;
  //int image_init;
  int k;
  _XMP_nodes_t *nodes;
  //MPI_Comm comm_coarray;

  if (dim <= 0 || corank < dim)
    _XMPCO_fatal("Too large or non-positive argument 'dim' of this_image:"
                 "%d\n", dim);

  nodes = cinfo->nodes;
  if (nodes != NULL) {
    image_coarray = _XMPCO_this_image_onNodes(nodes);
  } else {
    image_coarray = _XMPCO_get_currentThisImage();
  }

  if (image_coarray == 0)    // This image is out of the nodes.
    return 0;

  magic = image_coarray - 1;
  k = dim - 1;
  for (int i = 0; i < k; i++) {
    size = cinfo->cosize[i];
    magic /= size;
  }
  size = cinfo->cosize[k];
  index = magic % size;
  return index + cinfo->lcobound[k];
}


/*************************************************\
  values obtained from nodes
\*************************************************/

MPI_Comm _XMPCO_get_comm_of_nodes(_XMP_nodes_t *nodes)
{
  if (!nodes->is_member)
    return MPI_COMM_NULL;
  return *(MPI_Comm*)(nodes->comm);
}

int _XMPCO_num_images_onNodes(_XMP_nodes_t *nodes)
{
  return nodes->comm_size;
}

int _XMPCO_this_image_onNodes(_XMP_nodes_t *nodes)
{
  if (!nodes->is_member)
    return 0;   // This image is out of the node.
  return nodes->comm_rank + 1;
}


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
  _XMP_nodes_t *nodes;

  if ((nodes = _XMPCO_get_imageDirNodes()) == NULL)
    nodes = _XMP_get_execution_nodes();

  return nodes->comm_size;
}

int _XMPCO_get_currentThisImage()
{
  _XMP_nodes_t *nodes;

  if ((nodes = _XMPCO_get_imageDirNodes()) == NULL)
    nodes = _XMP_get_execution_nodes();

  return nodes->comm_rank + 1;
}

MPI_Comm _XMPCO_get_currentComm()
{
  MPI_Comm *commp;
  _XMP_nodes_t *nodes;

  if ((nodes = _XMPCO_get_imageDirNodes()) == NULL)
    nodes = _XMP_get_execution_nodes();

  commp = (MPI_Comm*)nodes->comm;
  return *commp;
}


BOOL _XMPCO_is_subset_exec()
{
  if (_XMPCO_get_currentNumImages() < _initialNumImages)
    // now executing in a task region
    return TRUE;
  if (_XMPCO_get_imageDirNodes() != NULL)
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
  ///////////////////////////////////////
  _XMPCO_debugPrint("gege comm1=%d, comm2=%d, image1=%d\n", comm1, comm2, image1);
  ///////////////////////////////////////

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
int _XMPCO_get_initial_image_withDescPtr(int image, CoarrayInfo_t *descPtr)
{
  if (descPtr == NULL)
    return _XMPCO_transImage_current2initial(image);

  MPI_Comm nodesComm =
    _XMPCO_get_comm_fromCoarrayInfo(descPtr);
  if (nodesComm == MPI_COMM_NULL)
    return _XMPCO_transImage_current2initial(image);

  // The coarray is specified with a COARRAY directive.

  int initImage =  _XMPCO_transImage_withComm(nodesComm, image,
                                             MPI_COMM_WORLD);

  _XMPCO_debugPrint("*** got the initial image (%d) from the image mapping to nodes (%d)\n",
                    initImage, image);

  return initImage;
}


/*****************************************\
  Image-directive nodes
\*****************************************/

void _XMPCO_clean_imageDirNodes()
{
  _imageDirNodes = NULL;
}

void _XMPCO_set_imageDirNodes(_XMP_nodes_t *nodes)
{
  if (_imageDirNodes != NULL)
    _XMP_fatal("INTERNAL: _imageDirNodes was not consumed but is defined.");
  _imageDirNodes = nodes;

  _XMPCO_debugPrint("SET _imageDirNodes (%d nodes) done.\n", nodes->comm_size);
}

_XMP_nodes_t *_XMPCO_get_imageDirNodes()
{
  return _imageDirNodes;
}

// get and clean
_XMP_nodes_t *_XMPCO_consume_imageDirNodes()
{
  _XMP_nodes_t *ret = _imageDirNodes;
  _imageDirNodes = NULL;
  return ret;
}


/*****************************************\
  Communicator
\*****************************************/

MPI_Comm _XMPCO_get_comm_current()
{
  if (_imageDirNodes != NULL)
    return *(MPI_Comm*)(_imageDirNodes->comm);

  MPI_Comm *commp = (MPI_Comm*)(_XMP_get_execution_nodes()->comm);
  if (commp != NULL)
    return *commp;
  return MPI_COMM_WORLD;
}

MPI_Comm _XMPCO_consume_comm_current()
{
  _XMP_nodes_t *imageNodes = _XMPCO_consume_imageDirNodes();
  if (imageNodes != NULL)
    return *(MPI_Comm*)(imageNodes->comm);

  MPI_Comm *commp = (MPI_Comm*)(_XMP_get_execution_nodes()->comm);
  if (commp != NULL)
    return *commp;
  return MPI_COMM_WORLD;
}

