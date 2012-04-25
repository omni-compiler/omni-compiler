#include "xmpf_internal.h"

/*
 * For xmpf, initialize all
 */

void xmpf_init_all__()
{
  _XMP_init();
}

void xmpf_finalize_all__()
{
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
  xmpf_dbg_printf("array : *(a->array_addr_p) = %p\n", *(a->array_addr_p));
}


void _XMP_pack_array_2_DOUBLE(double *buf_addr, double *src_addr,
			      int *l, int *u, int *s, unsigned long long *d)
{

  int src_lower0 = l[0]; int src_upper0 = u[0]; int src_stride0 = s[0];
  int src_lower1 = l[1]; int src_upper1 = u[1]; int src_stride1 = s[1];
  unsigned long long src_dim_acc1 = d[1];

  for (int j = src_lower1; j <= src_upper1; j += src_stride1) {
    double *addr = src_addr + (j * src_dim_acc1);

    for (int i = src_lower0; i <= src_upper0; i += src_stride0) {
      //xmpf_dbg_printf("(i,j) = (%d,%d), %f\n", i, j, addr[i]);
      *buf_addr = addr[i];
      buf_addr++;
    }
  }

  /*
  if (_XMP_world_rank == 1){
    for (int j = 0; j <= 513; j++){
      double *addr = src_addr + (j * src_dim_acc1);
      for (int i = 0; i <= 513; i++){
	xmpf_dbg_printf("a(%3d,%3d) = %f\n", i, j, addr[i]);
      }
    }
  }
  */
  
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
      //xmpf_dbg_printf("(i,j) = (%d,%d), %f\n", i, j, addr[i]);
    }
  }

  /*
  if (_XMP_world_rank == 0){
    for (int j = 0; j <= 513; j++){
      double *addr = src_addr + (j * src_dim_acc1);
      for (int i = 0; i <= 513; i++){
	xmpf_dbg_printf("a(%3d,%3d) = %f\n", i, j, addr[i]);
      }
    }
  }
  */

  /*
  if (_XMP_world_rank == 0){
    int i, j;
    double *addr;
    j = 512;
    addr = dst_addr + (j * dst_dim_acc1);
    i = 511;
    xmpf_dbg_printf("a(%3d,%3d) = %f\n", i, j, addr[i]);

    j = 512;
    addr = dst_addr + (j * dst_dim_acc1);
    i = 513;
    xmpf_dbg_printf("a(%3d,%3d) = %f\n", i, j, addr[i]);

    j = 511;
    addr = dst_addr + (j * dst_dim_acc1);
    i = 512;
    xmpf_dbg_printf("a(%3d,%3d) = %f\n", i, j, addr[i]);

    j = 513;
    addr = dst_addr + (j * dst_dim_acc1);
    i = 512;
    xmpf_dbg_printf("a(%3d,%3d) = %f\n", i, j, addr[i]);

  }
  */

}
