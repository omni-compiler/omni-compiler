#include "xmpf_internal.h"

/*************************************************\
  this_image/num_images for the initial state
\*************************************************/

int XMPF_initial_this_image, XMPF_initial_num_images;

void _XMPF_set_initial_this_image()
{
  int rank;

  if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != 0)
    _XMPF_coarrayFatal("INTERNAL ERROR: illegal node rank of mine");

  XMPF_initial_this_image = rank + 1;
}

void _XMPF_set_initial_num_images()
{
  int size;

  if (MPI_Comm_size(MPI_COMM_WORLD, &size) != 0)
    _XMPF_coarrayFatal("INTERNAL ERROR: illegal node size of COMM_WORLD");
  XMPF_initial_num_images = size;
}

int _XMPF_get_initial_this_image()
{
  return XMPF_initial_this_image;
}

int _XMPF_get_initial_num_images()
{
  return XMPF_initial_num_images;
}


/*************************************************\
  this_image/num_images for the current context
\*************************************************/

/* 'this image' in a task region is defined as (MPI rank + 1),
 * which is not always equal to node_num of XMP.
 */
int _XMPF_get_current_this_image()
{
  // for inside of task block
  if (xmp_num_nodes() < xmp_all_num_nodes()) {
    int rank;
    _XMP_nodes_t *nodes = _XMP_get_execution_nodes();
    if (MPI_Comm_rank(*((MPI_Comm*)nodes->comm), &rank) == 0) {
      // got rank for the current communicator successfully
      return rank + 1;
    }
  }

  // get this_image for the initial communicator
  return _XMPF_get_initial_this_image();
}

int _XMPF_get_current_num_images()
{
  return xmp_num_nodes();
}


/*****************************************\
  inquire functions
\*****************************************/

/* see in xmpf_coarray_alloc.c */


/*****************************************\
  transformation functions
\*****************************************/

/*  MPI_Comm_size() of the current communicator
 */
int num_images_(void)
{
  //_XMPF_checkIfInTask("NUM_IMAGES");

  return _XMPF_get_current_num_images();
}


/*  (MPI_Comm_rank() + 1) in the current communicator
 */
int this_image_(void)
{
  int image;

  //_XMPF_checkIfInTask("THIS_IMAGE");

  return _XMPF_get_current_this_image();
}


/*****************************************\
  sync all
  All arguments, which are described to prohibit overoptimization 
  of compiler, are ignored in this library.
\*****************************************/

static unsigned int _count_syncall = 0;

void xmpf_sync_all_(void)
{
  _XMPF_checkIfInTask("syncall nostat");

  _count_syncall += 1;

  int state = 0;
  xmp_sync_all(&state);

  _XMPF_coarrayDebugPrint("SYNCALL done (count:%d, stat=%d)\n",
                          _count_syncall, state);
}

/* entry for automatic syncall at the end of procedures
 */
void xmpf_sync_all_auto_(void)
{
  _XMPF_checkIfInTask("syncall nostat");

  _count_syncall += 1;

  int state = 0;
  xmp_sync_all(&state);

  _XMPF_coarrayDebugPrint("SYNCALL_AUTO done (count:%d, stat=%d)\n",
                          _count_syncall, state);
}

void xmpf_sync_all_stat_core_(int *stat, char *msg, int *msglen)
{
  _XMPF_checkIfInTask("syncall with stat");

  static BOOL firstCall = TRUE;
  if (firstCall) {
    firstCall = FALSE;
    fprintf(stderr, "not supported yet: "
            "stat= specifier in SYNCALL statement\n");
    fprintf(stderr, "  -- ignored.\n");
  }

  int state;
  xmp_sync_all(&state);
}


/*****************************************\
  sync memory
\*****************************************/

void xmpf_sync_memory_nostat_(void)
{
  _XMPF_checkIfInTask("syncmemory nostat");

  int state;
  xmp_sync_memory(&state);

  _XMPF_coarrayDebugPrint("SYNC MEMORY done (stat=%d)\n",
                          state);
}

void xmpf_sync_memory_stat_(int *stat, char *msg, int *msglen)
{
  _XMPF_checkIfInTask("syncmemory with stat");

  static BOOL firstCall = TRUE;
  if (firstCall) {
    firstCall = FALSE;
    fprintf(stderr, "Warning: "
            "STAT= specifier of SYNC MEMORY is not supported yet and ignored.\n");
  }

  int state;
  xmp_sync_memory(&state);

  _XMPF_coarrayDebugPrint("SYNC MEMORY (with stat) done (stat=%d)\n",
                          state);
}


/* dummy function to supress compiler optimization
 * usage: in a Fortran program:
 *  call xmpf_touch(<any_variable_name> ...)
 */
void xmpf_touch_(void)
{
}


/*****************************************\
  sync images
\*****************************************/

void xmpf_sync_image_nostat_(int *image)
{
  int state;

  if (*image <= 0)
    _XMPF_coarrayFatal("ABORT: illegal image number (%d) found in SYNC IMAGES\n",
                       *image);

  _XMPF_coarrayDebugPrint("SYNC IMAGES(image=%d) starts...\n", *image);
  xmp_sync_image(*image, &state);
  _XMPF_coarrayDebugPrint("SYNC IMAGES(image=%d) ends. (stat=%d)\n",
                          *image, state);
}

void xmpf_sync_image_stat_(int *image, int *stat, char *msg, int *msglen)
{
  static BOOL firstCall = TRUE;
  if (firstCall) {
    firstCall = FALSE;
    fprintf(stderr, "Warning: "
            "STAT= specifier of SYNC IMAGES is not supported yet and ignored.\n");
  }

  _XMPF_checkIfInTask("syncimage with stat");
  xmpf_sync_image_nostat_(image);
}


void xmpf_sync_images_nostat_(int *images, int *size)
{
  int state;

  _XMPF_coarrayDebugPrint("SYNC IMAGES (1-to-N) starts...\n");
  xmp_sync_images(*size, images, &state);
  _XMPF_coarrayDebugPrint("SYNC IMAGES (1-to-N) ends. (stat=%d)\n", state);
}

void xmpf_sync_images_stat_(int *images, int *size, int *stat,
                            char *msg, int *msglen)
{
  static BOOL firstCall = TRUE;
  if (firstCall) {
    firstCall = FALSE;
    fprintf(stderr, "Warning: "
            "STAT= specifier of SYNC IMAGES is not supported yet and ignored.\n");
  }

  _XMPF_checkIfInTask("syncimage with stat");
  xmpf_sync_images_nostat_(images, size);
}


void xmpf_sync_allimages_nostat_(void)
{
  int state;

  _XMPF_coarrayDebugPrint("SYNC IMAGES (1-to-ALL) starts...\n");
  xmp_sync_images_all(&state);
  _XMPF_coarrayDebugPrint("SYNC IMAGES (1-to-ALL) ends. (stat=%d)\n",
                          state);
}

void xmpf_sync_allimages_stat_(int *stat, char *msg, int *msglen)
{
  static BOOL firstCall = TRUE;
  if (firstCall) {
    firstCall = FALSE;
    fprintf(stderr, "Warning: "
            "STAT= specifier of SYNC IMAGES is not supported yet and ignored.\n");
  }

  _XMPF_checkIfInTask("syncimage with stat");
  xmpf_sync_allimages_nostat_();
}



/*****************************************\
  error message to reply to Fortran (temporary)
  (not used yet)
\*****************************************/

char *_XMPF_errmsg = NULL;

void xmpf_get_errmsg_(unsigned char *errmsg, int *msglen)
{
  int i, len;

  if (_XMPF_errmsg == NULL) {
    len = 0;
  } else {
    len = strlen(_XMPF_errmsg);
    if (len > *msglen)
      len = *msglen;
    memcpy(errmsg, _XMPF_errmsg, len);      // '\n' is not needed
  }

  for (i = len; i < *msglen; )
    errmsg[i++] = ' ';

  return;
}

  
