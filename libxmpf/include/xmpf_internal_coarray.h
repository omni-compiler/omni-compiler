/******************************************\
    internal header for COARRAY Fortran
\******************************************/
#define BOOL   int
#define TRUE   1
#define FALSE  0

// GET/PUT Interface types
// (see also XcodeML-Exc-Tools/src/exc/xmpF/XMPcoindexObj.java)
//#define GET_INTERFACE_TYPE 6           // varid last implementation
#define GET_INTERFACE_TYPE 8
//#define PUT_INTERFACE_TYPE 7           // varid last implementation
#define PUT_INTERFACE_TYPE 8

#if defined(_XMP_FJRDMA)
#  define ONESIDED_BOUNDARY ((size_t)4)
#  define ONESIDED_COMM_LAYER "FJRDMA"
#elif defined(_XMP_GASNET)
#  define ONESIDED_BOUNDARY ((size_t)1)
#  define ONESIDED_COMM_LAYER "GASNET"
#elif defined(_XMP_MPI3_ONESIDED)
#  define ONESIDED_BOUNDARY ((size_t)1)
#  define ONESIDED_COMM_LAYER "MPI3_ONESIDED"
#else
#  define ONESIDED_BOUNDARY ((size_t)1)
#  define ONESIDED_COMM_LAYER "(something unknown)"
#endif

#define ROUND_UP(n,p)         (((((size_t)(n))-1)/(p)+1)*(p))
#define ROUND_UP_BOUNDARY(n)  ROUND_UP((n),ONESIDED_BOUNDARY)

#define MALLOC_UNIT  ((size_t)4)
#define ROUND_UP_UNIT(n)      ROUND_UP((n),MALLOC_UNIT)

/*-- parameters --*/
#define DESCR_ID_MAX   250

extern int _XMP_boundaryByte;     // communication boundary (bytes)

/*-- codes --*/
#define COARRAY_GET_CODE  700
#define COARRAY_PUT_CODE  701

/* xmpf_coarray.c */
extern void _XMPF_coarray_init(void); 
extern void _XMPF_coarray_finalize(void); 

extern int _XMPF_get_coarrayMsg(void);
extern void _XMPF_set_coarrayMsg(int sw);
extern void _XMPF_reset_coarrayMsg(void);
extern unsigned XMPF_get_poolThreshold(void);
extern size_t XMPF_get_localBufSize(void);
extern BOOL XMPF_isSafeBufferMode(void);

extern void xmpf_coarray_msg_(int *sw);

extern char *_XMPF_errmsg;   // to answer ERRMSG argument in Fortran
extern void xmpf_copy_errmsg_(char *errmsg, int *msglen);

extern int _XMPF_nowInTask(void);   // for restriction check
extern void _XMPF_checkIfInTask(char *msgopt);   // restriction check
extern void _XMPF_coarrayDebugPrint(char *format, ...);
extern void xmpf_coarray_fatal_with_len_(char *msg, int *msglen);
extern void _XMPF_coarrayFatal(char *format, ...);

extern void xmpf_this_image_coarray_(void **descPtr, int *corank, int image[]);
extern int xmpf_this_image_coarray_dim_(void **descPtr, int *corank, int *dim);

/* xmpf_coarray_alloc.c */
typedef struct _coarrayInfo_t CoarrayInfo_t;

extern void xmpf_coarray_malloc_(void **descPtr, char **crayPtr,
                                 int *count, int *element, void **tag);
extern void xmpf_coarray_free_(void **descPtr);

extern void xmpf_coarray_malloc_pool_(void);
extern void xmpf_coarray_alloc_static_(void **descPtr, char **crayPtr,
                                       int *count, int *element,
                                       char *name, int *namelen);
extern void xmpf_coarray_regmem_static_(void **descPtr, void **baseAddr,
                                        int *count, int *element,
                                        char *name, int *namelen);
extern void xmpf_coarray_count_size_(int *count, int *element);

extern void xmpf_coarray_prolog_(void **tag, char *name, int *namelen);
extern void xmpf_coarray_epilog_(void **tag);

extern void xmpf_coarray_find_descptr_(void **descPtr, char *baseAddr,
                                       void **tag, char *name, int *namelen);
extern void xmpf_coarray_set_coshape_(void **descPtr, int *corank, ...);
extern void xmpf_coarray_set_varname_(void **descPtr, char *name, int *namelen);

extern int xmpf_coarray_get_image_index_(void **descPtr, int *corank, ...);

extern int xmpf_coarray_allocated_bytes_(void);
extern int xmpf_coarray_garbage_bytes_(void);

extern char *_XMPF_get_coarrayName(void *descPtr);
extern void *_XMPF_get_coarrayDesc(void *descPtr);
extern size_t _XMPF_get_coarrayOffset(void *descPtr, char *baseAddr);
extern void *_XMPF_get_localBufCoarrayDesc(char **baseAddr, size_t *offset,
                                           char **name);
extern void *_XMPF_get_coarrayDescFromAddr(char *localAddr, char **orgAddr,
                                           size_t *offset, char **nameAddr);
extern MPI_Comm _XMPF_get_communicatorFromDescPtr(void *descPtr);


extern void _XMPF_coarray_set_nodes(CoarrayInfo_t *cinfo, _XMP_nodes_t *nodes);
//extern _XMP_nodes_t *_XMPF_coarray_get_nodes(CoarrayInfo_t *cinfo);

/* xmpf_coarray_lib.c */
extern int xmpf_num_images_(void);
extern int xmpf_this_image_noargs_(void);
//extern int xmpf_num_nodes_(void);
//extern int xmpf_node_num_(void);
extern void xmpf_get_comm_current_(int *comm);

extern int XMPF_initial_this_image, XMPF_initial_num_images;
extern void _XMPF_set_this_image_initial(void);
extern void _XMPF_set_num_images_initial(void);
extern int _XMPF_this_image_initial(void);
extern int _XMPF_num_images_initial(void);

extern BOOL _XMPF_is_subset_exec(void);
extern MPI_Comm _XMPF_get_comm_current(void);
extern int _XMPF_this_image_current(void);
extern int _XMPF_num_images_current(void);
extern int _XMPF_transImage_current2initial(int image);
extern int _XMPF_get_initial_image_withDescPtr(int image, void *descPtr);

extern MPI_Comm _XMPF_get_comm_onNodes(_XMP_nodes_t *nodes);
extern int _XMPF_num_images_onNodes(_XMP_nodes_t *nodes);
extern int _XMPF_this_image_onNodes(_XMP_nodes_t *nodes);

extern int _XMPF_transImage_withComm(MPI_Comm comm1, int image1, MPI_Comm comm2);

extern void xmpf_sync_all_(void);
extern void xmpf_sync_all_auto_(void);
extern void xmpf_sync_all_stat_core_(int *stat, char *msg, int *msglen);

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
#if PUT_INTERFACE_TYPE==8
extern void xmpf_coarray_put_scalar_(void **descPtr, char **baseAddr, int *element,
                                     int *coindex, char *rhs);
extern void xmpf_coarray_put_array_(void **descPtr, char **baseAddr, int *element,
                                    int *coindex, char **rhsAddr, int *rank,
                                    int skip[], int skip_rhs[], int count[]);
extern void xmpf_coarray_put_spread_(void **descPtr, char **baseAddr, int *element,
                                     int *coindex, char *rhs, int *rank,
                                     int skip[], int count[]);
#else
extern void xmpf_coarray_put_scalar_(void **descPtr, char **baseAddr, int *element,
                                     int *coindex, char *rhs, int *condition);
extern void xmpf_coarray_put_array_(void **descPtr, char **baseAddr, int *element,
                                    int *coindex, char *rhs, int *condition,
                                    int *rank, ...);
extern void xmpf_coarray_put_spread_(void **descPtr, char **baseAddr, int *element,
                                     int *coindex, char *rhs, int *condition,
                                     int *rank, ...);
#endif
extern void _XMPF_coarrayInit_put(void);


/* xmpf_coarray_get.c */
extern void xmpf_coarray_get_scalar_(void **descPtr, char **baseAddr, int *element,
                                     int *coindex, char *result);
#if GET_INTERFACE_TYPE==8
extern void xmpf_coarray_get_array_(void **descPtr, char **baseAddr, int *element,
                                    int *coindex, char *result, int *rank,
                                    int skip[], int count[]);
#else
extern void xmpf_coarray_get_array_(void **descPtr, char **baseAddr, int *element,
                                    int *coindex, char *result, int *rank, ...);
#endif
extern void _XMPF_coarrayInit_get(void);


/* xmpf_async.c */
#ifdef _XMP_MPI3
extern _Bool xmp_is_async();
#endif

