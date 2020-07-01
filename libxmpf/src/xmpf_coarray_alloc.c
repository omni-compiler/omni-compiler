#include <string.h>
#include <stdarg.h>
#include "xmpf_internal_coarray.h"
#include "_xmpco_alloc.h"


static char* _to_Nth(int n);

/***********************************************\
  hidden inquire functions
\***********************************************/

int xmpf_coarray_malloc_bytes_()
{
  return (int)xmp_coarray_malloc_bytes();
}

int xmpf_coarray_allocated_bytes_()
{
  return (int)xmp_coarray_allocated_bytes();
}

int xmpf_coarray_garbage_bytes_()
{
  return (int)xmp_coarray_garbage_bytes();
}


/***********************************************\
  Allocation
\***********************************************/

/* construct descPtr only if needed
 */
void xmpf_coarray_malloc_(void **descPtr, char **crayPtr,
                          int *count, int *element, void **tag)
{
  CoarrayInfo_t* cinfo =
    XMPCO_malloc_coarray(crayPtr, *count, (size_t)(*element),
                         (*tag) ? (ResourceSet_t*)(*tag) : NULL);
  *descPtr = (void*)cinfo;

  // SYNCALL_AUTO
  //xmpf_sync_all_auto_();
}


void xmpf_coarray_regmem_(void **descPtr, void *var,
                          int *count, int *element, void **tag)
{
  CoarrayInfo_t* cinfo =
    XMPCO_regmem_coarray(var, *count, (size_t)(*element),
                         (*tag) ? (ResourceSet_t*)(*tag) : NULL);
  *descPtr = (void*)cinfo;

  // SYNCALL_AUTO
  //xmpf_sync_all_auto_();
}


void xmpf_coarray_alloc_static_(void **descPtr, char **crayPtr,
                                int *count, int *element,
                                int *namelen, char *name)
{
  CoarrayInfo_t* cinfo =
    XMPCO_malloc_staticCoarray(crayPtr, *count, (size_t)(*element),
                               *namelen, name);
  *descPtr = (void*)cinfo;
}

void xmpf_coarray_regmem_static_(void **descPtr, void **baseAddr,
                                 int *count, int *element,
                                 int *namelen, char *name)
{
  CoarrayInfo_t* cinfo =
    XMPCO_regmem_staticCoarray(*baseAddr, *count, (size_t)(*element),
                               *namelen, name);
  *descPtr = (void*)cinfo;
}


/***********************************************\
  Deallocation/Deregistration
  - to keep the reverse order of allocation,
    freeing memory is delayed until garbage collection.
\***********************************************/

void xmpf_coarray_free_(void **descPtr)
{
  CoarrayInfo_t *cinfo = (CoarrayInfo_t*)(*descPtr);
  XMPCO_free_coarray(cinfo);
}

void xmpf_coarray_deregmem_(void **descPtr)
{
  CoarrayInfo_t *cinfo = (CoarrayInfo_t*)(*descPtr);
  XMPCO_deregmem_coarray(cinfo);
}


/*****************************************\
  Initialization for memory pool
\*****************************************/

void xmpf_coarray_malloc_pool_()
{
  XMPCO_malloc_pool();

  // init library internal
  _XMPCO_coarrayInit_get();
  _XMPCO_coarrayInit_getsub();
  _XMPCO_coarrayInit_put();
}

void xmpf_coarray_count_size_(int *count, int *element)
{
  XMPCO_count_size(*count, (size_t)(*element));
}


/*****************************************\
  Prologue/epilogue code for each procedure
\*****************************************/

void xmpf_coarray_prolog_(void **tag, int *namelen, char *name)
{
  XMPCO_prolog((ResourceSet_t**)tag, *namelen, name);
}


void xmpf_coarray_epilog_(void **tag)
{
  XMPCO_epilog((ResourceSet_t**)tag);
}



/*****************************************\
   entries
\*****************************************/

/** generate and return a descriptor for a coarray DUMMY ARGUMENT
 *   1. find the memory chunk that contains the coarray data object,
 *   2. generate coarrayInfo for the coarray dummy argument and link it 
 *      to the memory chunk, and
 *   3. return coarrayInfo as descPtr
 */
void xmpf_coarray_find_descptr_(void **descPtr, char *addr,
                                int *namelen, char *name)
{
  *descPtr = (void*)XMPCO_find_descptr(addr, *namelen, name);
}


/*****************************************\
  set attributes of CoarrayInfo
\*****************************************/

/* translate m.
 * set the current lower and upper cobounds
 */
void xmpf_coarray_set_corank_(void **descPtr, int *corank)
{
  CoarrayInfo_t *cp = (CoarrayInfo_t*)(*descPtr);
  _XMPCO_set_corank(cp, *corank);
}


void xmpf_coarray_set_codim_(void **descPtr, int *dim, int *lb, int *ub)
{
  CoarrayInfo_t *cp = (CoarrayInfo_t*)(*descPtr);
  _XMPCO_set_codim_withBounds(cp, *dim, *lb, *ub);
}


/* translate n.
 * set the name of coarray object
 */
void xmpf_coarray_set_varname_(void **descPtr, int *namelen, char *name)
{
  CoarrayInfo_t *cp = (CoarrayInfo_t*)(*descPtr);
  _XMPCO_set_varname(cp, *namelen, name);
}



/***********************************************\
  ENTRY: for COARRAY directive
   set XMP descriptor of the corresponding nodes
\***********************************************/

/* construct descPtr if needed
 */
void xmpf_coarray_set_nodes_(void **descPtr, void **nodesDesc)
{
  CoarrayInfo_t *cinfo = (CoarrayInfo_t*)(*descPtr);
  _XMP_nodes_t *nodes = (_XMP_nodes_t*)(*nodesDesc);
  cinfo = _XMPCO_set_nodes(cinfo, nodes);
  *descPtr = (void*)cinfo;
}


/***********************************************\
  ENTRY: for IMAGE directive
   set the nodes specified with IMAGE directive
\***********************************************/

void xmpf_coarray_set_image_nodes_(void **nodesDesc)
{
  _XMP_nodes_t *nodes = (_XMP_nodes_t*)(*nodesDesc);
  _XMPCO_set_imageDirNodes(nodes);
}


/***********************************************\
  ENTRY
   inquire function this_image(coarray)
   inquire function this_image(coarray, dim)
\***********************************************/

int xmpf_this_image_coarray_dim_(void **descPtr, int *corank, int *dim)
{
  return XMPCO_this_image_coarray_dim((CoarrayInfo_t*)(*descPtr), *corank, *dim);
}

void xmpf_this_image_coarray_(void **descPtr, int *corank, int image[])
{
  XMPCO_this_image_coarray((CoarrayInfo_t*)(*descPtr), *corank, image);
}


/***********************************************\
  ENTRY
   inquire function lcobound/ucobound(coarray)
   inquire function lcobound/ucobound(coarray ,dim)
\***********************************************/

void lcobound_(void)
{
  _XMPF_coarrayFatal("INTERNAL ERROR: illegal call of lcobound_");
}

void ucobound_(void)
{
  _XMPF_coarrayFatal("INTERNAL ERROR: illegal call of ucobound_");
}


int xmpf_cobound_dim_(void **descPtr, int *dim, int *kind,
                      int *lu, int *corank)
{
  int index;
  int k = *dim - 1;

  if (*kind != 4)
    _XMP_fatal("Only kind=4 is allowed in lcobound/ucobound.");

  if (k < 0 || *corank <= k)
    _XMP_fatal("Argument 'dim' of lcobound/ucobound is out of range");

  CoarrayInfo_t *cinfo = (CoarrayInfo_t*)(*descPtr);

  if (*lu <= 0)
    index = cinfo->lcobound[k];
  else
    index = cinfo->ucobound[k];

  return index;
}

void xmpf_cobound_nodim_subr_(void **descPtr, int *kind, 
                              int *lu, int *corank, int bounds[])
{
  if (*kind != 4)
    _XMP_fatal("Only kind=4 is allowed in lcobound/ucobound.");

  CoarrayInfo_t *cinfo = (CoarrayInfo_t*)(*descPtr);

  for (int i = 0; i < *corank; i++) {
    if (*lu <= 0)
      bounds[i] = cinfo->lcobound[i];
    else
      bounds[i] = cinfo->ucobound[i];
  }
}

/*  other interface for internal use
 */
int xmpf_lcobound_(void **descPtr, int *dim)
{
  CoarrayInfo_t *cinfo = (CoarrayInfo_t*)(*descPtr);
  return cinfo->lcobound[*dim - 1];
}

int xmpf_ucobound_(void **descPtr, int *dim)
{
  CoarrayInfo_t *cinfo = (CoarrayInfo_t*)(*descPtr);
  return cinfo->ucobound[*dim - 1];
}


/***********************************************\
  ENTRY
   inquire function image_index(coarray, sub)
\***********************************************/

void image_index_(void)
{
  _XMPF_coarrayFatal("INTERNAL ERROR: illegal call of image_index_");
}


int xmpf_image_index_(void **descPtr, int coindexes[])
{
  int i, idx, lb, ub, factor, count, image;

  CoarrayInfo_t *cp = (CoarrayInfo_t*)(*descPtr);

  count = 0;
  factor = 1;
  for (i = 0; i < cp->corank; i++) {
    idx = coindexes[i];
    lb = cp->lcobound[i];
    ub = cp->ucobound[i];
    if (idx < lb) {
      _XMPF_coarrayFatal("The %s cosubscript of coarray \'%s\' is too small.\n"
                         "  value=%d, range=[%d,%d]\n",
                         _to_Nth(i+1), cp->name, idx, lb, ub);
      return 0;
    }
    if (ub < idx && i < cp->corank - 1) {
      _XMPF_coarrayFatal("The %s cosubscript of coarray \'%s\' is too large.\n"
                         "  value=%d, range=[%d,%d]\n",
                         _to_Nth(i+1), cp->name, idx, lb, ub);
      return 0;
    }
    count += (idx - lb) * factor;
    factor *= cp->cosize[i];
  }

  image = count + 1;
  if (image > _XMPCO_get_currentNumImages())
    image = 0;

  return image;
}


/*  another interface for internal use
 */
int xmpf_coarray_get_image_index_(void **descPtr, int *corank, ...)
{
  int i, idx, lb, ub, factor, count;
  va_list(args);
  va_start(args, corank);

  CoarrayInfo_t *cp = (CoarrayInfo_t*)(*descPtr);

  if (cp->corank != *corank) {
    _XMPF_coarrayFatal("INTERNAL: found corank %d, which is "
                       "different from the declared corank %d",
                       *corank, cp->corank);
  }

  count = 0;
  factor = 1;
  for (i = 0; i < *corank; i++) {
    idx = *va_arg(args, int*);
    lb = cp->lcobound[i];
    ub = cp->ucobound[i];
    if (idx < lb || ub < idx) {
      _XMPF_coarrayFatal("%s cosubscript of \'%s\', %d, "
                         "is out of range %d to %d.\n",
                         _to_Nth(i+1), cp->name, idx, lb, ub);
    }
    count += (idx - lb) * factor;
    factor *= cp->cosize[i];
  }

  va_end(args);

  return count + 1;
}



/***********************************************\
   local
\***********************************************/

char* _to_Nth(int n)
{
  static char work[6];

  switch (n) {
  case 1:
    return "1st";
  case 2:
    return "2nd";
  case 3:
    return "3rd";
  case 21:
    return "21st";
  case 22:
    return "22nd";
  case 23:
    return "23rd";
  default:
    break;
  }

  sprintf(work, "%dth", n);
  return work;
}
