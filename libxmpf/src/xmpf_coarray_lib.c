#include "xmpf_internal.h"

/*****************************************\
  transformation functions
\*****************************************/

/*  MPI_Comm_size() of the current communicator
 */
int num_images_(void)
{
  _XMPF_checkIfInTask("num_images()");
  return xmp_num_nodes();
}

/*  (MPI_Comm_rank() + 1) in the current communicator
 */
int this_image_(void)
{
  _XMPF_checkIfInTask("this_image()");
  return xmp_node_num();
}


/*****************************************\
  sync all
\*****************************************/

void xmpf_sync_all_nostat_(void)
{
  static unsigned int id = 0;

  _XMPF_checkIfInTask("sync all");

  id += 1;

  if (_XMPF_coarrayMsg) {
    _XMPF_coarrayDebugPrint("SYNC ALL in (id=%d)\n", id);
  }

  int status;
  xmp_sync_all(&status);

  if (_XMPF_coarrayMsg) {
    _XMPF_coarrayDebugPrint("SYNC ALL out (id=%d)\n", id);
  }
}

void xmpf_sync_all_stat_(int *stat, char *msg, int *msglen)
{
  _XMPF_checkIfInTask("sync all");

  static BOOL firstCall = TRUE;
  if (firstCall) {
    firstCall = FALSE;
    fprintf(stderr, "not supported yet: "
            "stat= specifier in SYNC ALL statement\n");
    fprintf(stderr, "  -- ignored.\n");
  }

  int status;
  xmp_sync_all(&status);
}


/*****************************************\
  sync memory
\*****************************************/

void xmpf_sync_memory_nostat_(void)
{
  _XMPF_checkIfInTask("sync memory");

  int status;
  xmp_sync_memory(&status);
}

void xmpf_sync_memory_stat_(int *stat, char *msg, int *msglen)
{
  _XMPF_checkIfInTask("sync memory");

  static BOOL firstCall = TRUE;
  if (firstCall) {
    firstCall = FALSE;
    fprintf(stderr, "not supported yet: "
            "stat= specifier in SYNC MEMORY statement\n");
    fprintf(stderr, "  -- ignored.\n");
  }

  int status;
  xmp_sync_memory(&status);
}


/*****************************************\
  sync images
\*****************************************/

void xmpf_sync_image_nostat_(int *image)
{
  int status;
  xmp_sync_image(*image, &status);
}

void xmpf_sync_image_stat_(int *image, int *stat, char *msg, int *msglen)
{
  static BOOL firstCall = TRUE;
  if (firstCall) {
    firstCall = FALSE;
    fprintf(stderr, "not supported yet: "
            "stat= specifier in SYNC IMAGES (<image>) statement\n");
    fprintf(stderr, "  -- ignored.\n");
  }

  _XMPF_checkIfInTask("sync image");

  int status;
  xmp_sync_image(*image, &status);
}


void xmpf_sync_images_nostat_(int *images, int *size)
{
  int status;
  xmp_sync_images(*size, images, &status);
}

void xmpf_sync_images_stat_(int *images, int *size, int *stat,
                            char *msg, int *msglen)
{
  static BOOL firstCall = TRUE;
  if (firstCall) {
    firstCall = FALSE;
    fprintf(stderr, "not supported yet: "
            "stat= specifier in SYNC IMAGES (<images>) statement\n");
    fprintf(stderr, "  -- ignored.\n");
  }

  _XMPF_checkIfInTask("sync image");

  int status;
  xmp_sync_images(*size, images, &status);
}


void xmpf_sync_allimages_nostat_(void)
{
  int status;
  xmp_sync_images_all(&status);
}

void xmpf_sync_allimages_stat_(int *stat, char *msg, int *msglen)
{
  static BOOL firstCall = TRUE;
  if (firstCall) {
    firstCall = FALSE;
    fprintf(stderr, "not supported yet: "
            "stat= specifier in SYNC IMAGES (*) statement\n");
    fprintf(stderr, "  -- ignored.\n");
  }

  _XMPF_checkIfInTask("sync image");

  int status;
  xmp_sync_images_all(&status);
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

  





