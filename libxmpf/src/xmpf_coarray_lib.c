#include "xmpf_internal.h"

/*************************************************\
  INITIAL or ABSOLUTE images
\*************************************************/
static int _initial_this_image;
static int _initial_num_images;

void _XMPF_set_initial_this_image()
{
  int rank;

  if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != 0)
    _XMPF_coarrayFatal("INTERNAL ERROR: "
                       "MPI_Comm_rank(MPI_COMM_WORLD, ) failed");

  _initial_this_image = rank + 1;
}

void _XMPF_set_initial_num_images()
{
  int size;

  if (MPI_Comm_size(MPI_COMM_WORLD, &size) != 0)
    _XMPF_coarrayFatal("INTERNAL ERROR: "
                       "MPI_Comm_size(MPI_COMM_WORLD, ) failed");
  _initial_num_images = size;
}

int _XMPF_get_initial_this_image()
{
  return _initial_this_image;
}

int _XMPF_get_initial_num_images()
{
  return _initial_num_images;
}


/*************************************************\
  CURRENT or RELATIVE images and communicators
\*************************************************/
/*  see struct _XMP_nodes_type in libxmp/include/xmp_data_struct.h
 */

static int _translate_images(MPI_Comm comm1, int image1, MPI_Comm comm2);

BOOL _XMPF_is_subset_exec()
{
  return xmp_num_nodes() < _initial_num_images;
}

MPI_Comm _XMPF_get_current_comm()
{
  return *(MPI_Comm*)(_XMP_get_execution_nodes()->comm);
}

int _XMPF_get_current_num_images()
{
  return xmp_num_nodes();
}

/* 'this image' in a task region is defined as (MPI rank + 1),
 * which is not always equal to node_num of XMP in Language Spec V1.x.
 */
int _XMPF_get_current_this_image()
{
  if (_XMPF_is_subset_exec()) {
    int rank;
    if (MPI_Comm_rank(_XMPF_get_current_comm(), &rank) != 0)
      _XMPF_coarrayFatal("INTERNAL: MPI_Comm_rank failed");

    // got rank for the current communicator successfully
    return rank + 1;
  }

  // get this_image for the initial communicator
  return _XMPF_get_initial_this_image();
}

int _XMPF_get_initial_image(int image)
{
  if (_XMPF_is_subset_exec()) {
    _XMPF_coarrayDebugPrint("*** get initial image from image %d\n", image);

    return _translate_images(_XMPF_get_current_comm(), image,
                             MPI_COMM_WORLD);
  }
  return image;
}


/*  TEMPORARY VERSION
 *  It would be better using MPI_Group_translate_ranks with vector arguments.
 */
static void _get_initial_images(int size, int images1[], int images2[])
{
  MPI_Comm comm = _XMPF_get_current_comm();

  for (int i=0; i < size; i++)
    images2[i] = _translate_images(comm, images1[i], MPI_COMM_WORLD);
}


/*  TEMPORARY VERSION
 *  It would be better using MPI_Group_translate_ranks with vector arguments.
 */
static void _get_initial_allimages(int size, int images2[])
{
  int myImage = _XMPF_get_current_this_image();
  MPI_Comm comm = _XMPF_get_current_comm();
  int i,j;

  for (i=0, j=0; i < size + 1; i++) {
    if (i == myImage)
      continue;
    images2[j++] = _translate_images(comm, i, MPI_COMM_WORLD);
  }
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
int xmpf_num_images_(void)
{
  return _XMPF_get_current_num_images();
}


/*  (MPI_Comm_rank() + 1) in the current communicator
 */
int xmpf_this_image_noargs_(void)
{
  return _XMPF_get_current_this_image();
}


/*****************************************\
  sync all
  All arguments, which are described to prohibit overoptimization 
  of compiler, are ignored in this library.
\*****************************************/

static unsigned int _count_syncall = 0;
static int _sync_all_core();
static int _sync_all_withComm(MPI_Comm comm);

void xmpf_sync_all_(void)
{
  int stat;

  if (_XMPF_is_subset_exec()) {
    stat = _sync_all_withComm(_XMPF_get_current_comm());
    _XMPF_coarrayDebugPrint("SYNCALL withComm done (count:%d+, stat=%d)\n",
                            _count_syncall, stat);
    return;
  }

  stat = _sync_all_core();
  _XMPF_coarrayDebugPrint("SYNCALL done (count:%d, state=%d)\n",
                          _count_syncall, stat);
}

/* entry for automatic syncall at the end of procedures
 */
void xmpf_sync_all_auto_(void)
{
  int stat;

  if (_XMPF_is_subset_exec()) {
    stat = _sync_all_withComm(_XMPF_get_current_comm());
    _XMPF_coarrayDebugPrint("SYNCALL AUTO withComm done (count:%d+, stat=%d)\n",
                            _count_syncall, stat);
    return;
  } 

  stat = _sync_all_core();
  _XMPF_coarrayDebugPrint("SYNCALL_AUTO done (count:%d, stat=%d)\n",
                          _count_syncall, stat);
}


static int _sync_all_core()
{
  _count_syncall += 1;

  int state = 0;
  xmp_sync_all(&state);
  if (state != 0)
    _XMPF_coarrayFatal("SYNC ALL failed with state=%d", state);
  return state;
}


static int _sync_all_withComm(MPI_Comm comm)
{
  int state = 0;
  xmp_sync_memory(&state);
  if (state != 0)
    _XMPF_coarrayFatal("SYNC MEMORY inside SYNC ALL failed with state=%d",
                       state);
  state = MPI_Barrier(comm);
  if (state != 0)
    _XMPF_coarrayFatal("MPI_Barrier inside SYNC ALL failed with state=%d",
                       state);
  return state;
}



/* Error handling is not supported yet.
 * Simple xmpf_sync_all() is used instead.
 */
void xmpf_sync_all_stat_core_(int *stat, char *msg, int *msglen)
{
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
  int state;
  xmp_sync_memory(&state);

  _XMPF_coarrayDebugPrint("SYNC MEMORY done (stat=%d)\n",
                          state);
}

void xmpf_sync_memory_stat_(int *stat, char *msg, int *msglen)
{
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
 *  call xmpf_touch(<any_variable_name>, ...)
 */
void xmpf_touch_(void)
{
}


/*****************************************\
  sync images
\*****************************************/

void xmpf_sync_image_nostat_(int *image)
{
  int state = 0;
  int image0 = _XMPF_get_initial_image(*image);

  if (image0 <= 0)
    _XMPF_coarrayFatal("ABORT: illegal image number (%d) found in SYNC IMAGES",
                       image0);

  _XMPF_coarrayDebugPrint("SYNC IMAGES(image=%d) starts...\n", image0);
  xmp_sync_image(image0, &state);
  _XMPF_coarrayDebugPrint("SYNC IMAGES(image=%d) ends. (stat=%d)\n",
                          image0, state);
}

void xmpf_sync_images_nostat_(int *images, int *size)
{
  int state;

  int *images0 = (int*)malloc((sizeof(int)*(*size)));
  _get_initial_images(*size, images, images0);

  _XMPF_coarrayDebugPrint("SYNC IMAGES 1-to-N starts...\n");
  xmp_sync_images(*size, images0, &state);
  _XMPF_coarrayDebugPrint("SYNC IMAGES 1-to-N ends. (stat=%d)\n", state);

  free(images0);
}

void xmpf_sync_allimages_nostat_(void)
{
  int state;

  if (_XMPF_is_subset_exec()) {
    int size = _XMPF_get_current_num_images() - 1;    // #of images except myself
    int *images0 = (int*)malloc((sizeof(int)*size));
    _get_initial_allimages(size, images0);

    _XMPF_coarrayDebugPrint("SYNC IMAGES 1-to-SUBSET starts...\n");
    xmp_sync_images(size, images0, &state);
    _XMPF_coarrayDebugPrint("SYNC IMAGES 1-to-SUBSET ends. (stat=%d)\n", state);

    free(images0);
    return;
  } 

  _XMPF_coarrayDebugPrint("SYNC IMAGES 1-to-ALL starts...\n");
  xmp_sync_images_all(&state);
  _XMPF_coarrayDebugPrint("SYNC IMAGES 1-to-ALL ends. (stat=%d)\n",
                          state);
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

  
/*****************************************\
  tools 
\*****************************************/

int _translate_images(MPI_Comm comm1, int image1, MPI_Comm comm2)
{
  int image2, rank1, rank2;
  MPI_Group group1, group2;
  int stat1, stat2, stat3;

  rank1 = image1 - 1;
  stat1 = MPI_Comm_group(comm1, &group1);
  stat2 = MPI_Comm_group(comm2, &group2);
  //                 (in:Group1, n, rank1[n], Group2, out:rank2[n])
  stat3 = MPI_Group_translate_ranks(group1, 1, &rank1, group2, &rank2);
  image2 = rank2 + 1;

  if (stat1 != 0 || stat2 != 0 || stat3 != 0)
    _XMPF_coarrayFatal("INTERNAL: _translate_images failed with "
                       "stat1=%d, stat2=%d, stat3=%d",
                       stat1, stat2, stat3);

  _XMPF_coarrayDebugPrint("***IMAGE NUMBER translated from %d to %d\n",
                          image1, image2);

  return image2;
}

