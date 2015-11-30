#include "xmp_internal.h"
#include <string.h>

_XMP_async_comm_t _XMP_async_comm_tab[_XMP_ASYNC_COMM_SIZE] = { {0, 0, NULL, NULL, NULL} };

#if defined(OMNI_TARGET_CPU_KCOMPUTER) && defined(K_RDMA_REFLECT)
static void _XMP_wait_async_rdma(_XMP_async_comm_t *async);
#endif

_XMP_async_comm_t *_XMP_get_async(int async_id);
void _XMP_pop_async(int async_id);

_Bool is_async = false;
int _async_id;

extern void (*_XMP_unpack_comm_set)(void *recvbuf, int recvbuf_size,
				    _XMP_array_t *a, _XMP_comm_set_t *comm_set[][_XMP_N_MAX_DIM]);

//
//
//

static void _XMP_finalize_async_gmove(_XMP_async_gmove_t *gmove){

  // Unpack
  (*_XMP_unpack_comm_set)(gmove->recvbuf, gmove->recvbuf_size, gmove->a, gmove->comm_set);

  // Deallocate temporarls
  _XMP_free(gmove->sendbuf);
  _XMP_free(gmove->recvbuf);

  int n_exec_nodes = _XMP_get_execution_nodes()->comm_size;

  for (int rank = 0; rank < n_exec_nodes; rank++){
    for (int adim = 0; adim < gmove->a->dim; adim++){
      free_comm_set(gmove->comm_set[rank][adim]);
    }
  }

  _XMP_free(gmove->comm_set);

}

void _XMP_wait_async__(int async_id)
{
  _XMP_async_comm_t *async;

  //xmp_dbg_printf("async_id = %d\n", async_id);
  //if (!(async = _XMP_get_async(async_id))) _XMP_fatal("wrong async-id");
  if (!(async = _XMP_get_async(async_id))) return;

  int nreqs = async->nreqs;;
  MPI_Request *reqs = async->reqs;

#if defined(OMNI_TARGET_CPU_KCOMPUTER) && defined(K_RDMA_REFLECT)
  // For RDMA reflects, async->nreqs > 0 and async->reqs == NULL.
  if (nreqs && !reqs){
    _XMP_wait_async_rdma(async);
    return;
  }
#endif

  _XMP_TSTART(t0);
  MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);
  _XMP_TEND(xmptiming_.t_wait, t0);

  //
  // for asynchronous gmove
  //

  _XMP_async_gmove_t *gmove = async->gmove;
  if (gmove) _XMP_finalize_async_gmove(gmove);

  _XMP_pop_async(async_id);

}


// NOTE: xmp_test_async is defined in the spec and invoked in both XMP/C and XMP/F.
// So it must be moved to xmp_lib.c and xmpf_lib.c
int xmp_test_async_(int *async_id)
{
  _XMP_async_comm_t *async;

  //if (!(async = _XMP_get_async(*async_id))) _XMP_fatal("wrong async-id");
  if (!(async = _XMP_get_async(*async_id))) return 1;

  int nreqs = async->nreqs;
  MPI_Request *reqs = async->reqs;

#if defined(OMNI_TARGET_CPU_KCOMPUTER) && defined(K_RDMA_REFLECT)
  // For RDMA reflects, async->nreqs > 0 and async->reqs == NULL.
  _XMP_fatal("xmp_test_async not supported for RDMA.");
  /* if (nreqs && !reqs){ */
  /*   _XMP_test_async_rdma(async); */
  /*   return; */
  /* } */
#endif

  int flag;
  MPI_Testall(nreqs, reqs, &flag, MPI_STATUSES_IGNORE);

  if (flag){

    _XMP_async_gmove_t *gmove = async->gmove;
    if (gmove) _XMP_finalize_async_gmove(gmove);

    _XMP_pop_async(*async_id);

    return 1;

  }
  else {
    return 0;
  }

}


#if defined(OMNI_TARGET_CPU_KCOMPUTER) && defined(K_RDMA_REFLECT)

static void _XMP_wait_async_rdma(_XMP_async_comm_t *async)
{
  int nreqs = async->nreqs;
  int async_id = async->async_id;

  _XMP_async_comm_t *async1;

  struct FJMPI_Rdma_cq cq;

  _XMP_TSTART(t0);

  while (nreqs){

    while (FJMPI_Rdma_poll_cq(FJMPI_RDMA_NIC0, &cq) != FJMPI_RDMA_NOTICE);
    if (cq.tag == async_id){
      nreqs--;
    }
    else {
      //      if (!(async1 = _XMP_get_async(cq.tag))) _XMP_fatal("wrong async-id");
      async1 = _XMP_get_or_create_async(cq.tag);
      async1->nreqs--;
    }

    while (FJMPI_Rdma_poll_cq(FJMPI_RDMA_NIC1, &cq) != FJMPI_RDMA_NOTICE);
    if (cq.tag == async_id){
      nreqs--;
    }
    else {
      //      if (!(async1 = _XMP_get_async(cq.tag))) _XMP_fatal("wrong async-id");
      async1 = _XMP_get_or_create_async(cq.tag);
      async1->nreqs--;
    }

  }

  _XMP_TEND(xmptiming_.t_wait, t0);

  _XMP_pop_async(async_id);

  xmp_barrier();

}

#endif


//
// for Asynchronous Communication
//

/* void _XMP_set_async(int nreqs, MPI_Request *reqs, int async_id) */
/* { */
/*   int hash = async_id % _XMP_ASYNC_COMM_SIZE; */
/*   _XMP_async_comm_t *async = &_XMP_async_comm_tab[hash]; */

/*   if (async->nreqs == 0){ */
/*     async->async_id = async_id; */
/*     async->nreqs = nreqs; */
/*     async->reqs = reqs; */
/*     async->next = NULL; */
/*   } */
/*   else { */
/*     while (async->next) async = async->next; */
/*     _XMP_async_comm_t *new_async = _XMP_alloc(sizeof(_XMP_async_comm_t)); */
/*     new_async->async_id = async_id; */
/*     new_async->nreqs = nreqs; */
/*     new_async->reqs = reqs; */
/*     new_async->next = NULL; */
/*     async->next = new_async; */
/*   } */

/* } */


_XMP_async_comm_t *_XMP_get_async(int async_id)
{
  int hash = async_id % _XMP_ASYNC_COMM_SIZE;
  _XMP_async_comm_t *async = &_XMP_async_comm_tab[hash];

  //xmp_dbg_printf("get: hash = %d\n", hash);
  //xmp_dbg_printf("async->nreqs = %d\n", async->nreqs);

  if (async->nreqs != 0){
    if (async->async_id == async_id){
      return async;
    }
    else {
      while (async->next){
	async = async->next;
	if (async->async_id == async_id){
	  return async;
	}
      }
    }

  }

  return NULL;

}


_XMP_async_comm_t *_XMP_get_or_create_async(int async_id)
{
  int hash = async_id % _XMP_ASYNC_COMM_SIZE;
  _XMP_async_comm_t *async = &_XMP_async_comm_tab[hash];

  //xmp_dbg_printf("put: hash = %d\n", hash);

  if (async->nreqs != 0){
    if (async->async_id == async_id){
      return async;
    }
    else {

      while (async->next){
	async = async->next;
	if (async->async_id == async_id){
	  return async;
	}
      }

      async->next = _XMP_alloc(sizeof(_XMP_async_comm_t));
      async = async->next;

    }

  }

  async->async_id = async_id;
  async->nreqs = 0;
  async->reqs = _XMP_alloc(sizeof(MPI_Request) * _XMP_MAX_ASYNC_REQS);
  async->gmove = NULL;
  async->next = NULL;

  return async;

}


void _XMP_pop_async(int async_id)
{
  int hash = async_id % _XMP_ASYNC_COMM_SIZE;
  _XMP_async_comm_t *async = &_XMP_async_comm_tab[hash];

  // The case no comm. registered for async_id 0 and _XMP_pop_async called for 
  // async_id == 0, may occur and is inconsistent.
  // But, actually, the code below works without problems in such a case, even if
  // no hash == 0.
  if (async->async_id == async_id){

    if (async->next){
      _XMP_async_comm_t *t = async->next;
      async->async_id = t->async_id;
      async->nreqs = t->nreqs;
      async->reqs = t->reqs;
      async->next = t->next;
      _XMP_free(t->gmove);
      _XMP_free(t);
    }
    else {
      async->nreqs = 0;
      _XMP_free(async->reqs);
      _XMP_free(async->gmove);
      async->gmove = NULL;
    }

    return;

  }
  else {

    _XMP_async_comm_t *prev = async;

    while ((async = prev->next)){

      if (async->async_id == async_id){
	prev->next = async->next;
	_XMP_free(async->reqs);
	_XMP_free(async->gmove);
	_XMP_free(async);
	return;
      }
      
      prev = async;

    }

  }

  _XMP_fatal("internal error: inconsistent async table");

}


#ifdef _XMP_MPI3

void xmpc_init_async(int async_id){
  is_async = true;
  _async_id = async_id;
}


void xmpc_start_async(int async_id){
  is_async = false;
}

#endif
