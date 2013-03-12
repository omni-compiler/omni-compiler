#ifndef _XMP_ATOMIC_H
#define _XMP_ATOMIC_H

#ifdef _XMP_COARRAY_GASNET
#define _XMP_LOCK_CHUNK 8       // for lock

typedef struct xmp_lock{
  int islocked;
  gasnet_hsl_t  hsl;
  int  wait_size;   /* How many elements in wait_list */
  int  wait_head;   /* Index for next dequeue */
  int  wait_tail;   /* Index for next enqueue */
  int *wait_list;   /* Circular queue of waiting threads */
} xmp_gasnet_lock_t;

typedef enum {
  _XMP_LOCKSTATE_WAITING = 300,    /* waiting for a reply */
  _XMP_LOCKSTATE_GRANTED,    /* lock attempt granted */
  _XMP_LOCKSTATE_FAILED,     /* lock attempt failed */
  _XMP_LOCKSTATE_HANDOFF,    /* unlock op complete--handoff in progress */
  _XMP_LOCKSTATE_DONE        /* unlock op complete */
} xmp_gasnet_lock_state_t;

extern void _xmp_gasnet_lock(_XMP_coarray_t*, int, int);
extern void _xmp_gasnet_unlock(_XMP_coarray_t*, int, int);
extern void _xmp_gasnet_do_lock(int, xmp_gasnet_lock_t*, int*);
extern void _xmp_gasnet_lock_initialize(xmp_gasnet_lock_t*, int);
extern void _xmp_gasnet_do_unlock(int, xmp_gasnet_lock_t*, int*, int*);
extern void _xmp_gasnet_do_lockhandoff(int);
extern void _xmp_gasnet_unpack(gasnet_token_t, const char*, const size_t, 
			       const int, const int, const int, const int);
extern void _xmp_gasnet_unpack_using_buf(gasnet_token_t, const int, const int, const int, const int);
extern void _xmp_gasnet_unpack_reply(gasnet_token_t, const int);
extern void _xmp_gasnet_pack(gasnet_token_t, const char*, const size_t, 
			     const int, const int, const int, const size_t, const int, const int);
extern void _xmp_gasnet_unpack_get_reply(gasnet_token_t, char *, size_t, const int, const int);

/* Every handler function needs a uniqe number between 200-255.   
 * The Active Message library reserves ID's 1-199 for itself: client libs must
 * use IDs between 200-255. 
 */
#define _XMP_GASNET_LOCK_REQUEST               200
#define _XMP_GASNET_SETLOCKSTATE               201
#define _XMP_GASNET_UNLOCK_REQUEST             202
#define _XMP_GASNET_LOCKHANDOFF                203
#define _XMP_GASNET_POST_REQUEST               204
#define _XMP_GASNET_UNPACK                     205
#define _XMP_GASNET_UNPACK_USING_BUF           206
#define _XMP_GASNET_UNPACK_REPLY               207
#define _XMP_GASNET_PACK                       208
#define _XMP_GASNET_UNPACK_GET_REPLY           209
#define _XMP_GASNET_PACK_USGIN_BUF             210
#define _XMP_GASNET_UNPACK_GET_REPLY_USING_BUF 211
#define _XMP_GASNET_PACK_GET_HANDLER           212
#define _XMP_GASNET_UNPACK_GET_REPLY_NONC      213
extern void _xmp_gasnet_lock_request(gasnet_token_t, int, uint32_t, uint32_t);
extern void _xmp_gasnet_setlockstate(gasnet_token_t, int);
extern void _xmp_gasnet_do_setlockstate(int);
extern void _xmp_gasnet_unlock_request(gasnet_token_t, int, uint32_t, uint32_t);
extern void _xmp_gasnet_lockhandoff(gasnet_token_t, int);
extern void _xmp_gasnet_post_request(gasnet_token_t, int, int);
extern void _xmp_gasnet_pack_using_buf(gasnet_token_t, const char*, const size_t,
				       const int, const int, const int, const int);
extern void _xmp_gasnet_unpack_get_reply_using_buf(gasnet_token_t);
extern void _xmp_gasnet_pack_get(gasnet_token_t, const char*, const size_t, const int,
				 const int, const int, const int, const size_t, const int, const int);
extern void _xmp_gasnet_unpack_get_reply_nonc(gasnet_token_t, char *, size_t, const int, const int, const int);

/*  Macros for splitting and reassembling 64-bit quantities  */
#define HIWORD(arg)     ((uint32_t) (((uint64_t)(arg)) >> 32))
#if PLATFORM_COMPILER_CRAY || PLATFORM_COMPILER_INTEL
/* workaround irritating warning #69: Integer conversion resulted in truncation.                                 
   which happens whenever Cray C or Intel C sees address-of passed to SEND_PTR                                   
*/
#define LOWORD(arg)     ((uint32_t) (((uint64_t)(arg)) & 0xFFFFFFFF))
#else
#define LOWORD(arg)     ((uint32_t) ((uint64_t)(arg)))
#endif
#define UPCRI_MAKEWORD(hi,lo) (   (((uint64_t)(hi)) << 32) \
                                  | (((uint64_t)(lo)) & 0xffffffff) )

/* These macros are referred from upcr.h of Berkeley UPC */
/*                                                                                                                 
 * Network polling                                                                                                 
 * ===============                                                                                                 
 *                                                                                                                 
 * The upcr_poll() function explicitly causes the runtime to attempt to make                                       
 * progress on any network requests that may be pending.  While many other                                         
 * runtime functions implicitly do this as well (i.e. most of those which call                                     
 * the network layer) this function may be useful in cases where a large amount                                    
 * of time has elapsed since the last runtime call (e.g. if a great deal of                                        
 * application-level calculation is taking place).  This function may also be                                      
 * indirectly when a upc_fence is used.                                                                            
 *                                                                                                                 
 * upcr_poll() also provides a null strict reference, corresponding to upc_fence in the                            
 * UPC memory model.                                                                                               
 * DOB: we should really rename upcr_poll to upcr_fence, but this would break                                      
 * compatibility between old runtimes and new translators, so until the next                                       
 * major runtime interface upgrade, (b)upc_poll expands to upcr_poll_nofence,                                      
 * which polls without the overhead of strict memory fences.                                                       
 */

/* Bug 2996 - upcr_poll_nofence should also yield in polite mode to get                                          
 * resonable performance from a spin-loop constructed according to our                                           
 * recommendations.                                                                                              
 * The bug was first seen w/ smp-conduit, but when using a network we                                            
 * cannot claim to know if gasnet_AMPoll() is going to yield or not.                                             
 * With an RMDA-capable transport one actually could expect that it                                              
 * would NOT.                                                                                                    
 */
#define upcr_poll_nofence() do {        \
    gasnet_AMPoll();			\
  } while (0)
#if GASNET_CONDUIT_SMP && !UPCRI_UPC_PTHREADS && !GASNET_PSHM
/* in the special case of exactly one UPC thread, nothing is required for                                        
 * correctness of fence (poll is likely a no-op as well, included solely                                         
 * for tracing purposes)                                                                                         
 */
#define upcr_poll() upcr_poll_nofence()
#else
/* in all other cases, a fence needs to act as a null strict reference,                                          
 * which means we need an architectural membar & optimization barrier to                                         
 * ensure that surrounding relaxed shared and local operations are not                                           
 * reordered in any way across this point (which could be visible if other                                       
 * CPU's or an RDMA enabled NIC are modifying memory via strict operations).                                     
 * We need both an WMB and RMB within the fence, but it doesn't actually matter                                  
 * whether they come before or after the optional poll (which is added as                                        
 * a performance optimization, to help ensure progress in spin-loops using fence).                               
 * We combine them in a call to gasnett_local_mb(), which on some architectures                                  
 * can be slightly more efficient than WMB and RMB called in sequence.                                           
 */
#define upcr_poll() do {              \
    gasnett_local_mb();               \
    upcr_poll_nofence();              \
  } while (0)
#endif

#define _xmp_lock_t xmp_gasnet_lock_t

#endif // _XMP_COARRAY_GASNET

extern void _xmp_lock(_XMP_coarray_t*, int, int);
extern void _xmp_unlock(_XMP_coarray_t*, int, int);
extern void _xmp_lock_initialize(_xmp_lock_t*, int);

extern void _XMP_post(int, int);
extern void _XMP_wait(int, int);
#endif

