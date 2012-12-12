#include "xmpf_internal.h"
#include "config.h"

//#define DBG 1

#ifdef DBG
extern double t_copy;
extern double t_comm;
extern double t_mem;
#endif

/*
 * For xmpf, initialize all
 */

void xmpf_init_all__()
{
  _XMP_init(0, NULL);
}

void xmpf_finalize_all__()
{
#ifdef DBG
  xmpf_dbg_printf("t_copy = %f\n", t_copy);
  xmpf_dbg_printf("t_comm = %f\n", t_comm);
  xmpf_dbg_printf("t_mem  = %f\n", t_mem);
#endif

  _XMP_finalize();
}


void xmpf_debug_()
{
  int flag = 0;
  MPI_Initialized(&flag);
  printf("xmpf_debug init flag=%d\n",flag);
  if (!flag) {
    MPI_Init(NULL, NULL);
  }
  printf("_XMP_world_size=%d, _XMP_world_rank=%d\n",
	 _XMP_world_size, _XMP_world_rank);
  {
    int myrank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    MPI_Comm_size(MPI_COMM_WORLD,&nproc);
    printf("nproc=%d myrank=%d\n", nproc, myrank);
  }
}

#include <string.h>

void xmpf_print_(char *msg, int l)
{
  char buf[512];
  strncpy(buf,msg,l);
  buf[l] = '\0';
  printf("[%d] %s\n",_XMP_world_rank, buf);
  MPI_Barrier(MPI_COMM_WORLD);
}


void xmpf_dbg_printf(char *fmt,...)
{
  char buf[512];
  va_list args;

  va_start(args,fmt);
  vsprintf(buf,fmt,args);
  va_end(args);

  printf("[%d] %s",_XMP_world_rank, buf);
  fflush(stdout);
}


void xmpf_array___(_XMP_array_t **a_desc)
{
  _XMP_array_t *a = *a_desc;
  xmpf_dbg_printf("array : a->array_addr_p = %p\n", a->array_addr_p);
}


void dumy(void)
{
  _XMP_pack_array(NULL, NULL, 0, 0, 0, NULL, NULL, NULL, NULL);
  _XMP_unpack_array(NULL, NULL, 0, 0, 0, NULL, NULL, NULL, NULL);
}


size_t _XMP_get_datatype_size(int datatype)
{
  size_t size;

  // size of each type is obtained from config.h.
  // Note: need to fix when building a cross compiler.
  switch (datatype){

  case _XMP_N_TYPE_BOOL:
    size = SIZEOF__BOOL; break;

  case _XMP_N_TYPE_CHAR:
  case _XMP_N_TYPE_UNSIGNED_CHAR:
    size = SIZEOF_UNSIGNED_CHAR; break;

  case _XMP_N_TYPE_SHORT:
  case _XMP_N_TYPE_UNSIGNED_SHORT:
    size = SIZEOF_UNSIGNED_SHORT; break;

  case _XMP_N_TYPE_INT:
  case _XMP_N_TYPE_UNSIGNED_INT:
    size = SIZEOF_UNSIGNED_INT; break;

  case _XMP_N_TYPE_LONG:
  case _XMP_N_TYPE_UNSIGNED_LONG:
    size = SIZEOF_UNSIGNED_LONG; break;

  case _XMP_N_TYPE_LONGLONG:
  case _XMP_N_TYPE_UNSIGNED_LONGLONG:
    size = SIZEOF_UNSIGNED_LONG_LONG; break;

  case _XMP_N_TYPE_FLOAT:
  case _XMP_N_TYPE_FLOAT_IMAGINARY:
    size = SIZEOF_FLOAT; break;

  case _XMP_N_TYPE_DOUBLE:
  case _XMP_N_TYPE_DOUBLE_IMAGINARY:
    size = SIZEOF_DOUBLE; break;

  case _XMP_N_TYPE_LONG_DOUBLE:
  case _XMP_N_TYPE_LONG_DOUBLE_IMAGINARY:
    size = SIZEOF_LONG_DOUBLE; break;

  case _XMP_N_TYPE_FLOAT_COMPLEX:
    size = SIZEOF_FLOAT * 2; break;

  case _XMP_N_TYPE_DOUBLE_COMPLEX:
    size = SIZEOF_DOUBLE * 2; break;

  case _XMP_N_TYPE_LONG_DOUBLE_COMPLEX:
    size = SIZEOF_LONG_DOUBLE * 2; break;

  case _XMP_N_TYPE_NONBASIC: // should be fixed for structures.
  default:
    size = 0; break;
  }

  return size;
}


/*
void _XMP_pack_array_2_DOUBLE(double *buf_addr, double *src_addr,
			      int *l, int *u, int *s, unsigned long long *d)
{

  int src_lower0 = l[0]; int src_upper0 = u[0]; int src_stride0 = s[0];
  int src_lower1 = l[1]; int src_upper1 = u[1]; int src_stride1 = s[1];
  unsigned long long src_dim_acc1 = d[1];

  for (int j = src_lower1; j <= src_upper1; j += src_stride1) {
    double *addr = src_addr + (j * src_dim_acc1);

    for (int i = src_lower0; i <= src_upper0; i += src_stride0) {
      *buf_addr = addr[i];
      buf_addr++;
    }
  }

}


void _XMP_unpack_array_2_DOUBLE(double *dst_addr, double *buf_addr,
				int *l, int *u, int *s, unsigned long long *d)
{
  int dst_lower0 = l[0]; int dst_upper0 = u[0]; int dst_stride0 = s[0];
  int dst_lower1 = l[1]; int dst_upper1 = u[1]; int dst_stride1 = s[1];
  unsigned long long dst_dim_acc1 = d[1];

  for (int j = dst_lower1; j <= dst_upper1; j += dst_stride1) {
    double *addr = dst_addr + (j * dst_dim_acc1);

    for (int i = dst_lower0; i <= dst_upper0; i += dst_stride0) {
      addr[i] = *buf_addr;
      buf_addr++;
    }
  }

}
*/
