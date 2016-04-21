#include "xmp_internal.h"

void _XMP_mpi_atomic_define(int target_rank, _XMP_coarray_t *c, size_t offset, int value, size_t elmt_size)
{
  if(elmt_size != sizeof(int)){
    _XMP_fatal("invalid element-size was specified for _XMP_mpi_atomic_define");
  }

  MPI_Win win = _XMP_mpi_coarray_get_window(c, /*is_acc*/false);
  MPI_Aint raddr = (MPI_Aint)( (char*)c->addr[target_rank] + elmt_size * offset );

  // MPI_Accumulate with MPI_REPLACE act as MPI_Put
  MPI_Accumulate(&value, 1, MPI_INT,
		 target_rank, raddr, 1, MPI_INT,
		 MPI_REPLACE, win);
  
  MPI_Win_flush_local(target_rank, win);
}

void _XMP_mpi_atomic_ref(int target_rank, _XMP_coarray_t *c, size_t offset, int *value, size_t elmt_size)
{
  if(elmt_size != sizeof(int)){
    _XMP_fatal("invalid element-size was specified for _XMP_mpi_atomic_ref");
  }

  MPI_Win win = _XMP_mpi_coarray_get_window(c, /*is_acc*/false);
  MPI_Aint raddr = (MPI_Aint)( (char*)c->addr[target_rank] + elmt_size * offset );

  // MPI_Fetch_and_op or MPI_Get_accumulate with MPI_NO_OP act as MPI_Get
  MPI_Fetch_and_op(NULL, value, MPI_INT,
		   target_rank, raddr,
		   MPI_NO_OP, win);

  /*
  MPI_Get_accumulate(NULL, elmt_size, MPI_BYTE,
		     value, elmt_size, MPI_BYTE,
		     target_rank, raddr, elmt_size, MPI_BYTE,
		     MPI_NO_OP, win);
  }
  */
  
  MPI_Win_flush_local(target_rank, win);
}
