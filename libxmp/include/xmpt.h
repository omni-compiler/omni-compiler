#include <limits.h>
#include <stdint.h>
#include "xmp_constant.h"

#define XMP_ASYNC_NONE INT_MIN

typedef void *xmp_desc_t;

typedef struct _xmpt_subscript_t {
  int ndims;                  /* number of dimensions */
  int omit;                   /* flag to tell whether an optional clause is omitted */
  int lbound[_XMP_N_MAX_DIM]; /* lower bound of the subscript */
  int ubound[_XMP_N_MAX_DIM]; /* upper bound of the subscript */
  int marker[_XMP_N_MAX_DIM]; /* mark periodic or step */
} *xmpt_subscript_t;

typedef enum {
  xmpt_gmove_collective = 0,
  xmpt_gmove_in = 1,
  xmpt_gmove_out = 2
} xmpt_gmove_kind_t;

typedef void *xmpt_tool_data_t;

typedef int xmpt_async_id_t;

// Coarray identifier
typedef uint64_t xmpt_coarray_id_t;

// Inquiry function for the global identifier of a coarray
extern int xmpt_coarray_get_gid(
  xmpt_coarray_id_t c
);

//
// Other inquiry functions for coarray identifiers (e.g. name)
// ...

/* typedef enum { */

/*   // */
/*   // 4: event may occur; callback always invoked when event occurs */
/*   // */
  
/*   xmpt_event_reflect_begin, */
/*   xmpt_event_reflect_begin_async, */
/*   xmpt_event_reflect_end, */

/*   xmpt_event_gmove_end, */

/*   xmpt_event_bcast_end, */

/*   // image control statements */
/*   xmpt_event_sync_memory_begin, */
/*   xmpt_event_sync_memory_end, */
/*   xmpt_event_sync_all_begin, */
/*   xmpt_event_sync_all_end, */
/*   xmpt_event_sync_images_begin, */
/*   xmpt_event_sync_images_end, */
  
/*   XMPT_EVENT_FULLY_SUPPORTED, */

/*   // */
/*   // 3: event may occur; callback invoked when convenient */
/*   // */
  
/*   xmpt_event_gmove_begin,         // needs support for scalars */
/*   xmpt_event_gmove_begin_async,   // needs support for scalars */
  
/*   xmpt_event_wait_async_begin,    // needs support for ON */
/*   xmpt_event_wait_async_end,      // needs support for ON */

/*   xmpt_event_bcast_begin,         // needs support for ON */
/*   xmpt_event_bcast_begin_async,   // needs support for ON */

/*   // coarray access */
/*   xmpt_event_coarray_remote_write, */
/*   xmpt_event_coarray_remote_read, */
/*   xmpt_event_coarray_local_write, */
/*   xmpt_event_coarray_local_read, */

/*   XMPT_EVENT_PARTIALLY_SUPPORTED, */

/*   // */
/*   // 2: event will never occur in this runtime */
/*   // */
  
/*   xmpt_event_tasks_begin, */
/*   xmpt_event_tasks_end, */

/*   xmpt_event_array_begin, */
/*   xmpt_event_array_end, */

/*   xmpt_event_sync_image_begin, */
/*   xmpt_event_sync_image_end, */
/*   xmpt_event_sync_images_all_begin, */
/*   xmpt_event_sync_images_all_end, */
  
/*   XMPT_EVENT_NEVER, */

/*   // */
/*   // 1: event may occur; no callback is possible (e.g., not yet implemented) */
/*   // */
  
/*   xmpt_event_task_begin, */
/*   xmpt_event_task_end, */

/*   xmpt_event_loop_begin, */
/*   xmpt_event_loop_end, */

/*   xmpt_event_barrier_begin, */
/*   xmpt_event_barrier_end, */

/*   xmpt_event_reduction_begin, */
/*   xmpt_event_reduction_begin_async, */
/*   xmpt_event_reduction_end, */
  
/*   XMPT_EVENT_ALL */
/* } xmpt_event_t; */

typedef enum xmpt_event_e {
  xmpt_event_task_begin            = 1,
  xmpt_event_task_end              = 2,
  xmpt_event_tasks_begin           = 3,
  xmpt_event_tasks_end             = 4,
  xmpt_event_loop_begin            = 5,
  xmpt_event_loop_end              = 6,
  xmpt_event_array_begin           = 7,
  xmpt_event_array_end             = 8,
  xmpt_event_reflect_begin         = 9,
  xmpt_event_reflect_begin_async   = 10,
  xmpt_event_reflect_end           = 11,
  xmpt_event_gmove_begin           = 12,
  xmpt_event_gmove_begin_async     = 13,
  xmpt_event_gmove_end             = 14,
  xmpt_event_barrier_begin         = 15,
  xmpt_event_barrier_end           = 16,
  xmpt_event_reduction_begin       = 17,
  xmpt_event_reduction_begin_async = 18,
  xmpt_event_reduction_end         = 19,
  xmpt_event_bcast_begin           = 20,
  xmpt_event_bcast_begin_async     = 21,
  xmpt_event_bcast_end             = 22,
  xmpt_event_wait_async_begin      = 23,
  xmpt_event_wait_async_end        = 24,

  // coarray access
  xmpt_event_coarray_remote_write  = 25,
  xmpt_event_coarray_remote_read   = 26,  
  xmpt_event_coarray_local_write   = 27,
  xmpt_event_coarray_local_read    = 28,

  // image control statements
  xmpt_event_sync_memory_begin     = 29,
  xmpt_event_sync_memory_end       = 30,
  xmpt_event_sync_all_begin        = 31,
  xmpt_event_sync_all_end          = 32,
  xmpt_event_sync_image_begin      = 33,
  xmpt_event_sync_image_end        = 34,
  xmpt_event_sync_images_all_begin = 35,
  xmpt_event_sync_images_all_end   = 36,
  xmpt_event_sync_images_begin     = 37,
  xmpt_event_sync_images_end       = 38,

  XMPT_EVENT_ALL
  
} xmpt_event_t;


//
// Type signatures for callbacks of XMP runtimes
//

typedef void (*xmpt_event_single_desc_begin_t) (
  xmp_desc_t desc,            /* descriptor for either nodes, template or array */
  xmpt_subscript_t subsc,     /* subscript specification */
  xmpt_tool_data_t* data      /* pointer to store tool specific data */
);

typedef void (*xmpt_event_single_desc_begin_async_t) (
  xmp_desc_t desc,             /* descriptor for either nodes, template or array */
  xmpt_subscript_t subsc,      /* subscript specification */
  xmpt_async_id_t async_id,    /* async-id */
  xmpt_tool_data_t* data       /* pointer to store tool specific data */
);

typedef void (*xmpt_event_gmove_begin_t) (
  xmp_desc_t lhs_array_desc,   /* descriptor for lhs of array assignment */
  xmpt_subscript_t lhs_subsc,  /* subscript for lhs */
  xmp_desc_t rhs_array_desc,   /* descriptor for rhs of array assignment */
  xmpt_subscript_t rhs_subsc,  /* subscript for rhs */
  xmpt_gmove_kind_t kind,      /* in/out/collective */
  xmpt_tool_data_t* data       /* pointer to store tool specific data */
);

typedef void (*xmpt_event_gmove_begin_async_t) (
  xmp_desc_t lhs_array_desc,   /* descriptor for lhs of array assignment */
  xmpt_subscript_t lhs_subsc,  /* subscript for lhs */
  xmp_desc_t rhs_array_desc,   /* descriptor for rhs of array assignment */
  xmpt_subscript_t rhs_subsc,  /* subscript for rhs */
  xmpt_gmove_kind_t kind,      /* in/out/collective */
  xmpt_async_id_t async_id,    /* async-id */
  xmpt_tool_data_t* data       /* pointer to store tool specific data */
);

typedef void (*xmpt_event_bcast_begin_t) (
  void* variable,              /* address of variable to be broadcasted */
  int size,                    /* size of the variable in bytes */
  xmp_desc_t from_desc,        /* descriptor for either nodes, template or array */
  xmpt_subscript_t from_subsc, /* subscript specification */
  xmp_desc_t on_desc,          /* descriptor for either nodes, template or array */
  xmpt_subscript_t on_subsc,   /* subscript specification */
  xmpt_tool_data_t* data       /* pointer to store tool specific data */
);

typedef void (*xmpt_event_bcast_begin_async_t) (
  void* variable,              /* address of variable to be broadcasted */
  int size,                    /* size of the variable in bytes */
  xmp_desc_t from_desc,        /* descriptor for either nodes, template or array */
  xmpt_subscript_t from_subsc, /* subscript specification */
  xmp_desc_t on_desc,          /* descriptor for either nodes, template or array */
  xmpt_subscript_t on_subsc,   /* subscript specification */
  xmpt_async_id_t async_id,    /* async-id */
  xmpt_tool_data_t* data       /* pointer to store tool specific data */
);

typedef void (*xmpt_event_wait_async_begin_t) (
  xmpt_async_id_t async_id,    /* async id to be wait */
  //  xmp_desc_t from_desc,        /* descriptor for either nodes, template or array */
  //  xmpt_subscript_t from_subsc, /* subscript specification */
  xmp_desc_t on_desc,          /* descriptor for either nodes, template or array */
  xmpt_subscript_t on_subsc,   /* subscript specification */
  xmpt_tool_data_t* data       /* pointer to store tool specific data */
);

typedef void (*xmpt_event_end_t) (
  xmpt_tool_data_t* data       /* pointer to store tool specific data */
);

typedef void (*xmpt_event_array_begin_t) (
  xmp_desc_t array_desc,       /* descriptor for array to be assigned */
  xmp_desc_t template_desc,    /* descriptor for template-ref */
  xmpt_subscript_t subsc,      /* subscript specification */
  xmpt_tool_data_t* data       /* pointer to store tool specific data */
);

typedef void (*xmpt_event_begin_t) (
  xmpt_tool_data_t* data       /* pointer to store tool specific data */
);

//
// Type signatures for callbacks of coarray access events
//

typedef void (*xmpt_event_coarray_remote_t)(
  xmpt_coarray_id_t c,
  xmpt_subscript_t subsc,
  xmpt_subscript_t cosubsc,
  xmpt_tool_data_t* data
);

typedef void (*xmpt_event_coarray_local_t)(
  xmpt_coarray_id_t c,
  xmpt_subscript_t subsc,
  xmpt_tool_data_t* data
);

typedef void (*xmpt_event_sync_images_begin_t)(
  int num,
  int *images,
  xmpt_tool_data_t* data
);

typedef void (*xmpt_callback_t);

int xmpt_set_callback(xmpt_event_t, xmpt_callback_t);

extern xmpt_callback_t xmpt_callback[XMPT_EVENT_ALL];
extern int xmpt_enabled;

extern xmp_desc_t on_desc;
extern struct _xmpt_subscript_t on_subsc;
extern xmp_desc_t from_desc;
extern struct _xmpt_subscript_t from_subsc;
