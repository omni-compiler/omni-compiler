#ifndef _XMP_IO_SYS
#define _XMP_IO_SYS

// --------------- including headers  --------------------------------
#include "xmp.h"
#include "xmp_data_struct.h"

// --------------- structures ----------------------------------------
typedef struct xmp_file_t {
    MPI_File   fh;
    MPI_Offset disp;
    char       is_append;
} xmp_file_t;

typedef struct xmp_range_t /* structure of array section */
{
  int dims;                /* number of dimensions     */
  int *lb;                 /* lower bound of array section (array of size dims) */
  int *ub;                 /* upper bound of array section (array of size dims) */
  int *step;               /* stride of array section (array of size dims) */
} xmp_range_t;

typedef void* xmp_array_t;

// --------------- functions -----------------------------------------
// xmp_io.c
extern xmp_file_t *xmp_fopen_all(const char*, const char*);
extern int        xmp_fclose_all(xmp_file_t*);
extern int        xmp_fseek(xmp_file_t*, long long, int);
extern int        xmp_fseek_shared_all(xmp_file_t*, long long, int);
extern long long  xmp_ftell(xmp_file_t*);
extern long long  xmp_ftell_shared(xmp_file_t*);
extern long long  xmp_file_sync_all(xmp_file_t*);
extern ssize_t    xmp_fread_all(xmp_file_t*, void*, size_t, size_t);

#ifdef ORIGINAL
extern ssize_t    xmp_fread_darray_all(xmp_file_t*, xmp_array_t, xmp_range_t*);
extern ssize_t    xmp_fwrite_darray_all(xmp_file_t*, xmp_array_t, xmp_range_t*);
#else /* RIST */
extern ssize_t    xmp_fread_darray_all(xmp_file_t*, xmp_desc_t, xmp_range_t*);
extern ssize_t    xmp_fwrite_darray_all(xmp_file_t*, xmp_desc_t, xmp_range_t*);
#endif

extern ssize_t    xmp_fwrite_all(xmp_file_t*, void*, size_t, size_t);
extern ssize_t    xmp_fread_shared(xmp_file_t*, void*, size_t, size_t);
extern ssize_t    xmp_fwrite_shared(xmp_file_t*, void*, size_t, size_t);
extern ssize_t    xmp_fread(xmp_file_t*, void*, size_t, size_t);
extern ssize_t    xmp_fwrite(xmp_file_t*, void*, size_t, size_t);

#ifdef ORIGINAL
extern int        xmp_file_set_view_all(xmp_file_t*, long long, xmp_array_t, xmp_range_t*);
#else /* RIST */
extern int        xmp_file_set_view_all(xmp_file_t*, long long, xmp_desc_t, xmp_range_t*);
#endif

extern int        xmp_file_clear_view_all(xmp_file_t*, long long);

#ifdef ORIGINAL
#else /* RIST */
extern xmp_range_t *xmp_allocate_range(int);
extern void xmp_set_range(xmp_range_t*, int, int, int, int);
extern void xmp_free_range(xmp_range_t*);
#endif

#endif // _XMP_IO_SYS
