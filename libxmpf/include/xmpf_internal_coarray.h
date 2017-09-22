/******************************************\
    internal header for COARRAY Fortran
\******************************************/

#ifndef XMPF_INTERNAL_COARRAY_H
#define XMPF_INTERNAL_COARRAY_H

#include "xmpco_internal.h"

#define _XMPF_coarrayDebugPrint	if (_XMPCO_get_isMsgMode()) __XMPF_coarrayDebugPrint


/*-- parameters --*/
#define DESCR_ID_MAX   250

extern int _XMP_boundaryByte;     // communication boundary (bytes)

/*-- codes --*/
#define COARRAY_GET_CODE  700
#define COARRAY_PUT_CODE  701

/* xmpf_coarray.c */
extern void _XMPF_coarray_init(void); 
extern void _XMPF_coarray_finalize(void); 
//extern int _XMPF_get_coarrayMsg(void);
//extern void _XMPF_set_coarrayMsg(int sw);
//extern void _XMPF_reset_coarrayMsg(void);
//extern unsigned XMPF_get_poolThreshold(void);
//extern size_t XMPF_get_localBufSize(void);
//extern BOOL XMPF_isSafeBufferMode(void);
//extern BOOL XMPF_isSyncPutMode(void);
//extern BOOL XMPF_isEagerCommMode(void);

/* hidden API */
extern void xmpf_coarray_msg_(int *sw);

extern char *_XMPF_errmsg;   // to answer ERRMSG argument in Fortran
extern void xmpf_copy_errmsg_(char *errmsg, int *msglen);

extern void __XMPF_coarrayDebugPrint(char *format, ...);
extern void xmpf_coarray_fatal_with_len_(char *msg, int *msglen);
extern void _XMPF_coarrayFatal(char *format, ...);

extern void xmpf_this_image_coarray_(void **descPtr, int *corank, int image[]);
extern int xmpf_this_image_coarray_dim_(void **descPtr, int *corank, int *dim);

/* xmpf_coarray_alloc.c */
extern void xmpf_coarray_malloc_(void **descPtr, char **crayPtr,
                                 int *count, int *element, void **tag);
extern void xmpf_coarray_regmem_(void **descPtr, void *var,
                                 int *count, int *element, void **tag);
extern void xmpf_coarray_free_(void **descPtr);
extern void xmpf_coarray_deregmem_(void **descPtr);

extern void xmpf_coarray_malloc_pool_(void);
extern void xmpf_coarray_alloc_static_(void **descPtr, char **crayPtr,
                                       int *count, int *element,
                                       int *namelen, char *name);
extern void xmpf_coarray_regmem_static_(void **descPtr, void **baseAddr,
                                        int *count, int *element,
                                        int *namelen, char *name);
extern void xmpf_coarray_count_size_(int *count, int *element);

extern void xmpf_coarray_prolog_(void **tag, int *namelen, char *name);
extern void xmpf_coarray_epilog_(void **tag);

extern void xmpf_coarray_find_descptr_(void **descPtr, char *baseAddr,
                                       int *namelen, char *name);
extern void xmpf_coarray_set_corank_(void **descPtr, int *corank);
extern void xmpf_coarray_set_codim_(void **descPtr, int *dim, int *lb, int *ub);
extern void xmpf_coarray_set_varname_(void **descPtr, int *namelen, char *name);

// for internal use
extern int xmpf_coarray_get_image_index_(void **descPtr, int *corank, ...);

extern int xmpf_coarray_malloc_bytes_(void);
extern int xmpf_coarray_allocated_bytes_(void);
extern int xmpf_coarray_garbage_bytes_(void);

// for COARRAY directive
extern void _XMPF_coarray_set_nodes(CoarrayInfo_t *cinfo, _XMP_nodes_t *nodes);
//extern _XMP_nodes_t *_XMPF_coarray_get_nodes(CoarrayInfo_t *cinfo);

// for IMAGE directive
extern void _XMPF_coarray_clean_image_nodes(void);
extern void _XMPF_coarray_set_image_nodes(_XMP_nodes_t *nodes);
extern _XMP_nodes_t *_XMPF_coarray_get_image_nodes(void);
extern _XMP_nodes_t *_XMPF_coarray_consume_image_nodes(void);


/* xmpf_coarray_lib.c */
extern int xmpf_num_images_current_(void);
extern int xmpf_this_image_current_(void);
extern void xmpf_get_comm_current_(MPI_Fint *fcomm);
extern void xmpf_consume_comm_current_(MPI_Fint *fcomm);

extern int XMPF_initial_this_image, XMPF_initial_num_images;

extern MPI_Comm _XMPF_get_comm_current(void);
extern MPI_Comm _XMPF_consume_comm_current(void);
extern int _XMPF_transImage_current2initial(int image);
extern int _XMPF_get_initial_image_withDescPtr(int image, void *descPtr);

extern MPI_Comm _XMPF_get_comm_onNodes(_XMP_nodes_t *nodes);
extern int _XMPF_num_images_onNodes(_XMP_nodes_t *nodes);
extern int _XMPF_this_image_onNodes(_XMP_nodes_t *nodes);

extern int _XMPF_transImage_withComm(MPI_Comm comm1, int image1, MPI_Comm comm2);

extern void xmpf_sync_all_(void);
extern void xmpf_sync_all_auto_(void);
extern void xmpf_sync_all_stat_core_(int *stat, char *msg, int *msglen);
extern void xmpf_sync_all_withcomm_(MPI_Fint *fcomm);

extern void xmpf_sync_memory_nostat_(void);
extern void xmpf_sync_memory_stat_(int *stat, char *msg, int *msglen);

extern void xmpf_sync_image_nostat_(int *image);
extern void xmpf_sync_image_stat_(int *image,
                                  int *stat, char *msg, int *msglen);
extern void xmpf_sync_images_nostat_(int *images, int *size);
extern void xmpf_sync_images_stat_(int *images, int *size,
                                   int *stat, char *msg, int *msglen);
extern void xmpf_sync_allimages_nostat_(void);
extern void xmpf_sync_allimages_stat_(int *stat, char *msg, int *msglen);

extern void xmpf_critical_(void);
extern void xmpf_end_critical_(void);

/* xmpf_coarray_put.c */
extern void xmpf_coarray_put_scalar_(void **descPtr, char **baseAddr, int *element,
                                     int *coindex, char *rhs, BOOL *synchronouns);
extern void xmpf_coarray_put_array_(void **descPtr, char **baseAddr, int *element,
                                    int *coindex, char **rhsAddr, int *rank,
                                    int skip[], int skip_rhs[], int count[],
                                    BOOL *synchronous);
extern void xmpf_coarray_put_spread_(void **descPtr, char **baseAddr, int *element,
                                     int *coindex, char *rhs, int *rank,
                                     int skip[], int count[],
                                     BOOL *synchronous);
extern void _XMPF_coarrayInit_put(void);

/* xmpf_coarray_get.c */
extern void xmpf_coarray_get_scalar_(void **descPtr, char **baseAddr, int *element,
                                     int *coindex, char *result);
extern void xmpf_coarray_get_array_(void **descPtr, char **baseAddr, int *element,
                                    int *coindex, char *result, int *rank,
                                    int skip[], int count[]);
extern void _XMPF_coarrayInit_get(void);

/* xmpf_coarray_getsub.c */
extern void xmpf_coarray_getsub_array_(void **descPtr, char **baseAddr, int *element,
                                       int *coindex, char **localAddr, int *rank,
                                       int skip[], int skip_local[], int count[]);
extern void _XMPF_coarrayInit_getsub(void);

// common
extern void _XMPF_getVector_DMA(void *descPtr, char *baseAddr, int bytes, int coindex,
                                void *descDMA, size_t offsetDMA, char *nameDMA);

extern void _XMPF_getVector_buffer(void *descPtr, char *baseAddr, int bytesRU, int coindex,
                                   char *result, int bytes);


/* xmpf_async.c */
#ifdef _XMP_MPI3
extern _Bool xmp_is_async();
#endif

#endif /* !XMPF_INTERNAL_COARRAY_H */
