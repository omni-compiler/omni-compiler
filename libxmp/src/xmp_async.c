#include "xmp_internal.h"
#include <string.h>
static _Bool _xmp_is_async = false;

extern void (*_XMP_unpack_comm_set)(void *recvbuf, int recvbuf_size,
				    _XMP_array_t *a, _XMP_comm_set_t *comm_set[][_XMP_N_MAX_DIM]);
static _XMP_async_comm_t _XMP_async_comm_tab[_XMP_ASYNC_COMM_SIZE];
static _XMP_async_comm_t *_tmp_async = NULL;
#if defined(OMNI_TARGET_CPU_KCOMPUTER) && defined(K_RDMA_REFLECT)
static void _XMP_wait_async_rdma(_XMP_async_comm_t *async);
#endif

/**************************************************************************************/
/* DESCRIPTION : This function is called in communication directive, and when         */
/*               _xmp_is_async is true, the communication is executed asynchronously. */
/* NOTE        : Is it between xmpc_init_async() and xmpc_start_async() ?             */
/*               While between them, _xmp_is_async must be true.                      */
/*               Otherwise, _xmp_is_async must be false.                              */
/**************************************************************************************/
_Bool xmp_is_async()
{
  return _xmp_is_async;
}

/*************************************************************************/
/* DESCRIPTION : Return async descriptor which has async_id of argument. */
/* ARGUMENT    : [IN] async_id : ID of async.                            */
/* RETRUN      : Async descriptor.                                       */
/*************************************************************************/
_XMP_async_comm_t* _XMP_get_async(int async_id)
{
  int hash_id = async_id % _XMP_ASYNC_COMM_SIZE;
  _XMP_async_comm_t *async = &_XMP_async_comm_tab[hash_id];
  
  if(async->async_id == async_id)  return async;

  while(async->next){
    async = async->next;
    if(async->async_id == async_id)
      return async;
  }

  return NULL;
}

/*************************************************************/
/* DESCRIPTION : Initialize _XMP_async_comm_tab[].           */
/* NOTE        : This function is called in _XMP_init() once */
/*************************************************************/
void _XMP_initialize_async_comm_tab()
{
  for(int i=0;i<_XMP_ASYNC_COMM_SIZE;i++){
    _XMP_async_comm_tab[i].nreqs   = 0;
    _XMP_async_comm_tab[i].nnodes  = 0;
    _XMP_async_comm_tab[i].is_used = false;
    _XMP_async_comm_tab[i].node    = NULL;
    _XMP_async_comm_tab[i].reqs    = NULL;
    _XMP_async_comm_tab[i].gmove   = NULL;
    _XMP_async_comm_tab[i].next    = NULL;
  }
}

static void _XMP_finalize_async_gmove(_XMP_async_gmove_t *gmove)
{
  // Unpack
  (*_XMP_unpack_comm_set)(gmove->recvbuf, gmove->recvbuf_size, gmove->a, gmove->comm_set);

  // Deallocate temporarls
  _XMP_free(gmove->sendbuf);
  _XMP_free(gmove->recvbuf);

  int n_exec_nodes = _XMP_get_execution_nodes()->comm_size;

  for(int rank = 0; rank < n_exec_nodes; rank++)
    for(int adim = 0; adim < gmove->a->dim; adim++)
      free_comm_set(gmove->comm_set[rank][adim]);

  _XMP_free(gmove->comm_set);
}

/*******************************************************************/
/* DESCRIPTION : Wait until completing asynchronous communication. */
/* ARGUMENT    : [IN] async_id : ID of async                       */
/*******************************************************************/
void _XMP_wait_async__(int async_id)
{
  _XMP_async_comm_t *async;

  if(!(async = _XMP_get_async(async_id))) return;

  int nreqs         = async->nreqs;
  MPI_Request *reqs = async->reqs;

#if defined(OMNI_TARGET_CPU_KCOMPUTER) && defined(K_RDMA_REFLECT)
  // For RDMA reflects, async->nreqs > 0 and async->reqs == NULL.
  if(nreqs && !reqs){
    _XMP_wait_async_rdma(async);
    return;
  }
#endif

  _XMP_async_gmove_t *gmove = async->gmove;

  if (!gmove || gmove->mode == _XMP_N_GMOVE_NORMAL){
    _XMP_TSTART(t0);
    MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);
    _XMP_TEND(xmptiming_.t_wait, t0);
  }

  //
  // for asynchronous gmove
  //

  if (gmove){
    if (gmove->mode == _XMP_N_GMOVE_NORMAL) _XMP_finalize_async_gmove(gmove);
#ifdef _XMP_MPI3_ONESIDED
    else {
      int status;
      // NOTE: the send_buf field is used for an improper purpose.
      _XMP_sync_images_COMM((MPI_Comm *)gmove->sendbuf, &status);
    }
#endif
  }

}

// NOTE: xmp_test_async is defined in the spec and invoked in both XMP/C and XMP/F.
// So it must be moved to xmp_lib.c and xmpf_lib.c
int xmp_test_async_(int *async_id)
{
  _XMP_async_comm_t *async;

  if (!(async = _XMP_get_async(*async_id))) return 1;

  int nreqs = async->nreqs;
  MPI_Request *reqs = async->reqs;

#if defined(OMNI_TARGET_CPU_KCOMPUTER) && defined(K_RDMA_REFLECT)
  // For RDMA reflects, async->nreqs > 0 and async->reqs == NULL.
  _XMP_fatal("xmp_test_async not supported for RDMA.");
#endif

  int flag;
  MPI_Testall(nreqs, reqs, &flag, MPI_STATUSES_IGNORE);

  if(flag){
    _XMP_async_gmove_t *gmove = async->gmove;
    if (gmove) _XMP_finalize_async_gmove(gmove);

    xmpc_end_async(*async_id);
    return 1;
  }
  else{
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
  //  _XMP_TSTART(t0);
  while (nreqs){
    while (FJMPI_Rdma_poll_cq(FJMPI_RDMA_NIC0, &cq) != FJMPI_RDMA_NOTICE);
    if (cq.tag == async_id){
      nreqs--;
    }
    else{
      async1 = _XMP_get_async(cq.tag);
      async1->nreqs--;
    }

    while (FJMPI_Rdma_poll_cq(FJMPI_RDMA_NIC1, &cq) != FJMPI_RDMA_NOTICE);
    if (cq.tag == async_id){
      nreqs--;
    }
    else{
      async1 = _XMP_get_async(cq.tag);
      async1->nreqs--;
    }
  }
  //  _XMP_TEND(xmptiming_.t_wait, t0);
  xmpc_end_async(async_id);
  xmp_barrier();
}
#endif

/******************************************************/
/* DESCRIPTION : Return a current async descriptor.   */
/* RETRUN      : Async descriptor.                    */
/* NOTE        : This function is used when executing */
/*               asynchronous communication.          */
/******************************************************/
_XMP_async_comm_t *_XMP_get_current_async()
{
  return _tmp_async;
}

/**********************************************************************/
/* DESCRIPTION : Set a variable _xmp_is_async to true and initialize  */
/*               an async descriptor. Moreover initialized async      */
/*               descriptor is saved to a static variable _tmp_async. */
/*               The _tmp_async is used in _XMP_get_current_async().  */
/* ARGUMENT    : [IN] async_id : ID of async                          */
/* NOTE        : While _xmp_is_async is true, communication is        */
/*               executed asynchronously.                             */
/**********************************************************************/
void xmpc_init_async(int async_id)
{
  _xmp_is_async = true;

  int hash_id = async_id % _XMP_ASYNC_COMM_SIZE;
  _XMP_async_comm_t *async = &_XMP_async_comm_tab[hash_id];

  if(!async->is_used){
    async->async_id = async_id;
    async->reqs     = _XMP_alloc(sizeof(MPI_Request) * _XMP_MAX_ASYNC_REQS);
    async->is_used  = true;
    _tmp_async      = async;
  }
  else{
    if(async->async_id == async_id){
      _tmp_async = async;
    }
    else{
      while(async->next){
	async = async->next;
	if(async->async_id == async_id){
	  _tmp_async = async;
	  return;
	}
      }

      // When above while-statement cannot find a async descriptor,
      // the following lines create a new async descriptor
      async->next     = _XMP_alloc(sizeof(_XMP_async_comm_t));
      async           = async->next;
      async->async_id = async_id;
      async->nreqs    = 0;
      async->nnodes   = 0;
      async->is_used  = true;
      async->node     = NULL;
      async->reqs     = _XMP_alloc(sizeof(MPI_Request) * _XMP_MAX_ASYNC_REQS);
      async->gmove    = NULL;
      async->next     = NULL;
      _tmp_async      = async;
    }
  }
}

/**************************************************************************/
/* DESCRIPTION : Registor node descriptor. The node descriptor is freed   */
/*               after wait_async directive.                              */
/* ARGUMENT    : [IN] n : Node descritor.                                 */
/* NOTE        : This function is called in _XMP_exec_task_NODES_PART().  */
/*               In other words, this function is called in communication */
/*               directives with "async" and "on" clauses.                */
/**************************************************************************/
void _XMP_nodes_dealloc_after_wait_async(_XMP_nodes_t* n)
{
  if(_tmp_async->nnodes >= _XMP_MAX_ASYNC_NODES) // Fix me
    _XMP_fatal("Too many nodes");

  if(_tmp_async->nnodes == 0)
    _tmp_async->node = _XMP_alloc(sizeof(_XMP_nodes_t*) * _XMP_MAX_ASYNC_NODES);
  
  _tmp_async->node[_tmp_async->nnodes] = n;
  _tmp_async->nnodes++;
}

/****************************************************************/
/* DESCRIPTION : Set a variable _xmp_is_async to false.         */
/* NOTE        : This function is called after the asynchronous */
/*               communication has issued.                      */
/****************************************************************/
void xmpc_start_async()
{
  _xmp_is_async = false;
  _tmp_async    = NULL;
}

/********************************************************************/
/* DESCRIPTION : Free an async descriptor and its node descriptor.  */
/* ARGUMENT    : [IN] async : Async descriptor.                     */
/* NOTE        : Before calling this function, only async->next may */
/*               be saved.                                          */
/********************************************************************/
static void initialize_async(_XMP_async_comm_t *async)
{
  for(int i=0;i<async->nnodes;i++)
    _XMP_finalize_nodes(async->node[i]);

  _XMP_free(async->node);  async->node  = NULL;
  _XMP_free(async->gmove); async->gmove = NULL;
  _XMP_free(async->reqs);  async->reqs  = NULL;
  
  async->nreqs   = 0;
  async->nnodes  = 0;
  async->is_used = false;
  async->next    = NULL;
}

/*******************************************************************************/
/* DESCRIPTION : Free async descriptor.                                        */
/* ARGUMENT    : [IN] async_id : ID of async                                   */
/* NOTE        : This function is called after wait_async directive.           */
/*               When using wait_async directive with on-cluase, a function    */
/*               _XMP_wait_async__() does not be called by all executing node. */
/*               However, this function is called by all executing node to     */
/*               free async descriptor.                                        */
/*******************************************************************************/
void xmpc_end_async(int async_id)
{
  int hash_id = async_id % _XMP_ASYNC_COMM_SIZE;
  _XMP_async_comm_t *async = &_XMP_async_comm_tab[hash_id];
  
  if(!async->is_used) return;

  // The case no comm. registered for async_id 0 and xmpc_end_async called for
  // async_id == 0, may occur and is inconsistent.
  // But, actually, the code below works without problems in such a case, even if
  // no hash_id == 0.
  if(async->async_id == async_id){
    if(async->next == NULL){
      initialize_async(async);
    }
    else{
      _XMP_async_comm_t *next = async->next;
      initialize_async(async);
      async->async_id = next->async_id;
      async->nreqs    = next->nreqs;
      async->nnodes   = next->nnodes;
      async->is_used  = next->is_used;
      async->node     = next->node;
      async->gmove    = next->gmove;
      async->reqs     = next->reqs;
      async->next     = next->next;
      _XMP_free(next);
    }
    return;
  }
  else{
    _XMP_async_comm_t *prev = async;
    while((async = prev->next)){
      if(async->async_id == async_id){
	prev->next = async->next;
	initialize_async(async);
	_XMP_free(async);
	return;
      }
      prev = async;
    }
  }
}

