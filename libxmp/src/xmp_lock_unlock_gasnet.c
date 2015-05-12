/* Lock algorithm of XMP is based on that of Bercley UPC. */
#ifndef MPI_PORTABLE_PLATFORM_H
#define MPI_PORTABLE_PLATFORM_H
#endif 

#include "xmp_internal.h"
#include "gasnet_tools.h"

void _xmp_gasnet_lock_initialize(xmp_gasnet_lock_t* lock, int number_of_elements){
  int i;

  for(i=0;i<number_of_elements;i++){
    gasnet_hsl_init(&((lock + i)->hsl));
    (lock + i)->islocked = _XMP_N_INT_FALSE;
    (lock + i)->wait_size = MIN(_XMP_LOCK_CHUNK, _XMP_world_size);
    (lock + i)->wait_list = malloc((lock + i)->wait_size * sizeof(int));
    (lock + i)->wait_head = 0;
    (lock + i)->wait_tail = 0;
  }
}

void _xmp_gasnet_do_lock(int target_node, xmp_gasnet_lock_t* lock, int *replystate){
  gasnet_hsl_lock(&(lock->hsl));
  if(lock->islocked == _XMP_N_INT_TRUE){
    /* add waiter to end of queue */
    lock->wait_list[lock->wait_tail++] = target_node;
    if(lock->wait_tail == lock->wait_size)
      lock->wait_tail = 0;       // The wait point move to head because of circular queue.

    /* If full, grow NOW to preserve property that head==tail only when empty */
    if(lock->wait_tail == lock->wait_head){
      int *old_list = lock->wait_list;
      int  old_head = lock->wait_head;
      int  old_size = lock->wait_size;
      int  leading = old_size - old_head;
      lock->wait_size = MIN(old_size + _XMP_LOCK_CHUNK, _XMP_world_size);
      lock->wait_list = malloc(lock->wait_size * sizeof(int));
      memcpy(lock->wait_list, old_list+old_head, leading*sizeof(int));
      memcpy(lock->wait_list+leading, old_list, old_head*sizeof(int));
      free(old_list);
      lock->wait_head = 0;
      lock->wait_tail = old_size;
    }

    *replystate = _XMP_LOCKSTATE_WAITING;

  } else{
    lock->islocked = _XMP_N_INT_TRUE;

    *replystate = _XMP_LOCKSTATE_GRANTED;
  }
  gasnet_hsl_unlock(&(lock->hsl));
}

volatile static int local_lockstate;
volatile static int local_handoffarg;

void _xmp_gasnet_lock(_XMP_coarray_t* c, int position, int target_node){
  xmp_gasnet_lock_t *lockaddr = (xmp_gasnet_lock_t *)(c->addr[target_node]) + position;
  if(target_node == _XMP_world_rank){
    _xmp_gasnet_do_lock(target_node, lockaddr, (int *)(&local_lockstate));
  }else{
    local_lockstate = _XMP_LOCKSTATE_WAITING;

    /* this memory barrier prevents a race against GASNet handler on "local_lockstate",
       and provides the wmb half of the memory barrier semantics required by xmp_lock()
       not required in local case - synchronous HSL critical section takes care of it */
    gasnett_local_wmb();

    // only supports 64 bits arch.
    gasnet_AMRequestShort3(target_node, _XMP_GASNET_LOCK_REQUEST, _XMP_world_rank, HIWORD(lockaddr), LOWORD(lockaddr));
  }

  GASNET_BLOCKUNTIL(local_lockstate != (int)_XMP_LOCKSTATE_WAITING);
}

void _xmp_gasnet_do_unlock(int target_node, xmp_gasnet_lock_t *lock, int *replystate, int *replyarg){
  gasnet_hsl_lock(&(lock->hsl));
  if(lock->wait_head != lock->wait_tail) {   /* someone waiting - handoff ownership */
    *replyarg = lock->wait_list[lock->wait_head++];
    if(lock->wait_head == lock->wait_size)      lock->wait_head = 0;

    *replystate = _XMP_LOCKSTATE_HANDOFF;
  }else{  /* nobody waiting - unlock */
    lock->islocked = _XMP_N_INT_FALSE;

    gasnet_hsl_unlock(&(lock->hsl));
    *replystate = _XMP_LOCKSTATE_DONE;
  }
  gasnet_hsl_unlock(&(lock->hsl));
}

void _xmp_gasnet_unlock(_XMP_coarray_t* c, int position, int target_node){
  xmp_gasnet_lock_t *lockaddr = (xmp_gasnet_lock_t *)(c->addr[target_node]) + position;
  if(target_node == _XMP_world_rank){
    upcr_poll();
    _xmp_gasnet_do_unlock(_XMP_world_rank, lockaddr, (int *)&local_lockstate, (int *)&local_handoffarg);
  } else{
    local_lockstate = _XMP_LOCKSTATE_WAITING;

    /* this memory barrier prevents a race against GASNet handler on "local_lockstate",
       and provides the wmb half of the memory barrier semantics required by xmp_unlock()
       not required in local case - synchronous HSL critical section takes care of it
    */
    gasnett_local_wmb();

    // only supports 64 bits arch.
    gasnet_AMRequestShort3(target_node, _XMP_GASNET_UNLOCK_REQUEST, _XMP_world_rank, HIWORD(lockaddr), LOWORD(lockaddr));
  }

  GASNET_BLOCKUNTIL(local_lockstate != (int)_XMP_LOCKSTATE_WAITING);

  if(local_lockstate == _XMP_LOCKSTATE_HANDOFF){
    /* tell the next locker that he acquired */
    int next_node = local_handoffarg;
    
    if(next_node == _XMP_world_rank){
      _xmp_gasnet_do_setlockstate(_XMP_LOCKSTATE_GRANTED);
    } else{
      gasnet_AMRequestShort1(next_node, _XMP_GASNET_SETLOCKSTATE, _XMP_LOCKSTATE_GRANTED);
    }
  } // end if (local_lockstate == XMP_LOCKSTATE_HANDOFF)
}

void _xmp_gasnet_lock_request(gasnet_token_t token, int node, uint32_t addr_hi, uint32_t addr_lo){
  int replystate;
  xmp_gasnet_lock_t *lockaddr = (xmp_gasnet_lock_t *)UPCRI_MAKEWORD(addr_hi, addr_lo);
  
  _xmp_gasnet_do_lock(node, lockaddr, &replystate);

  if(replystate != (int)_XMP_LOCKSTATE_WAITING){
    gasnet_AMReplyShort1(token, _XMP_GASNET_SETLOCKSTATE, replystate);
  }
}

void _xmp_gasnet_setlockstate(gasnet_token_t token, int state){
  _xmp_gasnet_do_setlockstate(state);
}

void _xmp_gasnet_do_setlockstate(int state){
  gasnett_local_wmb();  // prevent the compiler from reordering.
  local_lockstate = state;
}

void _xmp_gasnet_unlock_request(gasnet_token_t token, int node, uint32_t addr_hi, uint32_t addr_lo){
  int replystate, replyarg;
  xmp_gasnet_lock_t *lockaddr = (xmp_gasnet_lock_t *)UPCRI_MAKEWORD(addr_hi, addr_lo);

  _xmp_gasnet_do_unlock(node, lockaddr, &replystate, &replyarg);

  if (replystate == _XMP_LOCKSTATE_HANDOFF){
    gasnet_AMReplyShort1(token, _XMP_GASNET_LOCKHANDOFF, replyarg);
  }else{
    gasnet_AMReplyShort1(token, _XMP_GASNET_SETLOCKSTATE, replystate);
  }
}

void _xmp_gasnet_lockhandoff(gasnet_token_t token, int handoffarg){
  _xmp_gasnet_do_lockhandoff(handoffarg);
}

void _xmp_gasnet_do_lockhandoff(int handoffarg){
  local_handoffarg = handoffarg;
  gasnett_local_wmb();
  local_lockstate = (int)_XMP_LOCKSTATE_HANDOFF;
}
