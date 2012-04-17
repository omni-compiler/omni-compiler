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
