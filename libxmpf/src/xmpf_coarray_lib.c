#include "xmpf_internal_coarray.h"
#include "xmpco_internal.h"


/*************************************************\
  CURRENT images
\*************************************************/

/* entry
 */
void xmpf_get_comm_current_(MPI_Fint *fcomm)
{
  *fcomm = MPI_Comm_c2f(_XMPCO_get_comm_current());
}

void xmpf_consume_comm_current_(MPI_Fint *fcomm)
{
  *fcomm = MPI_Comm_c2f(_XMPCO_consume_comm_current());
}




/*************************************************\
  Translation of images
\*************************************************/
/*  get image2 of MPI communicatior comm2 corresponding to image1 of comm1.
 */
int _XMPF_transImage_withComm(MPI_Comm comm1, int image1, MPI_Comm comm2)
{
  return _XMPCO_transImage_withComm(comm1, image1, comm2);
}


/*************************************************\
  translation between initial and current images
\*************************************************/

int _XMPF_transImage_current2initial(int image)
{
  return _XMPCO_transImage_current2initial(image);
}

/*  get the initial image index corresponding to the image index
 *  of the nodes that the coarray is mapped to.
 */
int _XMPF_get_initial_image_withDescPtr(int image, void *descPtr)
{
  return _XMPCO_get_initial_image_withDescPtr(image, descPtr);
}



/*************************************************\
  internal
\*************************************************/

/*  TEMPORARY VERSION
 *  It would be better using MPI_Group_translate_ranks with vector arguments.
 */
static void _get_initial_image_vector(int size, int images1[], int images2[])
{
  MPI_Comm comm = _XMPCO_get_comm_current();

  for (int i=0; i < size; i++)
    images2[i] = _XMPCO_transImage_withComm(comm, images1[i], MPI_COMM_WORLD);
}


/*  TEMPORARY VERSION
 *  It would be better using MPI_Group_translate_ranks with vector arguments.
 */
static void _get_initial_allimages(int size, int images2[])
{
  int myImage = _XMPCO_get_currentThisImage();
  MPI_Comm comm = _XMPCO_get_comm_current();
  int i,j;

  for (i=0, j=0; i < size + 1; i++) {
    if (i == myImage)
      continue;
    images2[j++] = _XMPCO_transImage_withComm(comm, i, MPI_COMM_WORLD);
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
int xmpf_num_images_current_(void)
{
  return _XMPCO_get_currentNumImages();
}


/*  (MPI_Comm_rank() + 1) in the current communicator
 */
int xmpf_this_image_current_(void)
{
  return _XMPCO_get_currentThisImage();
}


/*****************************************\
  sync all
\*****************************************/

void xmpf_sync_all_()
{
  XMPCO_sync_all();
}

void xmpf_sync_all_auto_()
{
  XMPCO_sync_all_auto();
}


/* entry for pre- and post-syncall of mpi_allreduce and mpi_bcast
 */
void xmpf_sync_all_withcomm_(MPI_Fint *fcomm)
{
  MPI_Comm comm = MPI_Comm_f2c(*fcomm);
  XMPCO_sync_all_withComm(comm);
}


/* Error handling is not supported yet.
 * Simply, xmpf_sync_all() is used instead.
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

  XMPCO_sync_all();
}


/*****************************************\
  sync memory
\*****************************************/

void xmpf_sync_memory_(void)
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
  sync images -- 3-type interface
\*****************************************/

void xmpf_sync_image_nostat_(int *image)
{
  int state = 0;
  int image0 = _XMPCO_transImage_current2initial(*image);

  _XMPF_coarrayDebugPrint("SYNC IMAGES(image=%d) starts...\n", image0);
  xmp_sync_image(image0-1, &state);
  _XMPF_coarrayDebugPrint("SYNC IMAGES(image=%d) ends. (stat=%d)\n",
                          image0, state);
  _XMPCO_clean_imageDirNodes();
}

void xmpf_sync_images_nostat_(int *images, int *size)
{
  int state;

  int *images0 = (int*)malloc((sizeof(int)*(*size)));
  for(int i=0;i<*size;i++)
    images[i]--;
  
  _get_initial_image_vector(*size, images, images0);

  _XMPF_coarrayDebugPrint("SYNC IMAGES 1-to-N starts...\n");
  xmp_sync_images(*size, images0, &state);
  _XMPF_coarrayDebugPrint("SYNC IMAGES 1-to-N ends. (stat=%d)\n", state);

  free(images0);
  _XMPCO_clean_imageDirNodes();
}

void xmpf_sync_allimages_nostat_(void)
{
  int state;

  if (_XMPCO_is_subset_exec()) {
    int size = _XMPCO_get_currentNumImages() - 1;    // #of images except myself
    int *images0 = (int*)malloc((sizeof(int)*size));
    _get_initial_allimages(size, images0);
    for(int i=0;i<size;i++)
      images0[i]--;

    _XMPF_coarrayDebugPrint("SYNC IMAGES 1-to-SUBSET starts...\n");
    xmp_sync_images(size, images0, &state);
    _XMPF_coarrayDebugPrint("SYNC IMAGES 1-to-SUBSET ends. (stat=%d)\n", state);

    free(images0);
    _XMPCO_clean_imageDirNodes();
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

  _XMPCO_checkIfInTask("syncimage with stat");
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

  _XMPCO_checkIfInTask("syncimage with stat");
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

  _XMPCO_checkIfInTask("syncimage with stat");
  xmpf_sync_allimages_nostat_();
}


/*****************************************\
  sync images -- alternative interface
\*****************************************/

void xmpf_sync_images_(int *nimages, int *images)
{
  switch (*nimages) {
  case 1:
    xmpf_sync_image_nostat_(images);
    break;
  case 0:
    xmpf_sync_allimages_nostat_();
    break;
  default:
    xmpf_sync_images_nostat_(images, nimages);
    break;
  }
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

  
