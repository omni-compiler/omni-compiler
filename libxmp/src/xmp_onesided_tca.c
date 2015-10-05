#include "xmp_internal.h"
#include "tca-api.h"
#include "pthread.h"
#include <time.h>
#include <sys/time.h>

void _XMP_tca_comm_init();
void _XMP_tca_comm_finalize();
static void create_comm_thread();
static void destroy_comm_thread();

static pthread_mutex_t _XMP_tca_mutex = PTHREAD_MUTEX_INITIALIZER;
  //
  //PTHREAD_ERRORCHECK_MUTEX_INITIALIZER_NP;

static pthread_t comm_thread;
static char _run_flag = 0;

/**
   Initialize TCA
*/
void _XMP_tca_initialize(int argc, char **argv)
{
  if(_XMP_world_rank == 0){
    fprintf(stderr, "TCA Library Version = %s\n", TCA_LIB_VERSION);
  }

  if(_XMP_world_size > 16)
    _XMP_fatal("TCA reflect has been not implemented in 16 more than nodes.");

  _XMP_tca_lock();
  TCA_CHECK(tcaInit());

  //this is probably unnecessary
  TCA_CHECK(tcaDMADescInt_Init()); // Initialize Descriptor (Internal Memory) Mode
  
  _XMP_tca_comm_init();
  create_comm_thread();
  _XMP_tca_unlock();
}

/**
 Finalize TCA
*/
void _XMP_tca_finalize()
{
  _XMP_tca_lock();
  destroy_comm_thread();
  _XMP_tca_comm_finalize();
  _XMP_tca_unlock();
}

//#define _USE_NOTIFY_TO_WAIT_PACKET

const int _wait_slot_offset = 16;
const int _dmac_channel = 0;
#define RING_SIZE (8)
const int _handle_sendrecv_tag = 123; //適当
const int _packet_wait_tag = 31;
const int _psn_wait_tag = 32;
typedef struct packet_t{
  int tag;
  int data;
} packet_t;
typedef unsigned long long psn_t;

typedef struct tca_ring_buf_t{
  //PSN = Packet Sequence Number
  int remote_rank;
  psn_t *psn_pairs; //[l_send, l_recv, r_send, r_recv]
  packet_t *send_buffer;
  packet_t *recv_buffer;
  tcaHandle local_send_buffer_handle;
  tcaHandle remote_send_buffer_handle;
  tcaHandle local_recv_buffer_handle;
  tcaHandle remote_recv_buffer_handle;
  tcaDesc *buffer_desc[RING_SIZE];
  
  tcaHandle local_psn_pairs_handle;
  tcaHandle remote_psn_pairs_handle;
  tcaDesc *psn_desc;
  psn_t last_send_local_recv_psn;
} tca_ring_buf_t;

tca_ring_buf_t *_ring_bufs;

static void _XMP_tca_ring_buf_init(tca_ring_buf_t* ring_buf, int remote_rank)
{
  ring_buf->remote_rank = remote_rank;
  
  //alloc psn
  size_t psn_size = sizeof(psn_t);
  TCA_CHECK(tcaMalloc((void**)&ring_buf->psn_pairs, psn_size * 4, tcaMemoryCPU));
  //fprintf(stderr, "psn_pairs (%p)\n", ring_buf->psn_pairs);
  memset(ring_buf->psn_pairs, 0, psn_size * 4);

  //create psn handle
  TCA_CHECK(tcaCreateHandle(&(ring_buf->local_psn_pairs_handle), ring_buf->psn_pairs, psn_size * 4, tcaMemoryCPU));

  //exchange psn handle
  MPI_Sendrecv(&(ring_buf->local_psn_pairs_handle), sizeof(tcaHandle), MPI_BYTE, remote_rank, _handle_sendrecv_tag,
	       &(ring_buf->remote_psn_pairs_handle), sizeof(tcaHandle), MPI_BYTE, remote_rank, _handle_sendrecv_tag,
	       MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  {
    tcaDesc* desc = tcaDescNew();
    const int psn_wait_slot = _wait_slot_offset + RING_SIZE;
    TCA_CHECK(tcaDescSetMemcpy(desc,
			       &(ring_buf->remote_psn_pairs_handle), psn_size * 2,
			       &(ring_buf->local_psn_pairs_handle), 0,
			       psn_size * 2,
			       tcaDMAUseInternal | tcaDMAUseNotifyInternal | tcaDMANotifySelf | tcaDMANotify,
			       psn_wait_slot,
			       _psn_wait_tag));
    ring_buf->psn_desc = desc;
  }

  //clear last_send_local_recv_psn
  ring_buf->last_send_local_recv_psn = 0;
  
  //alloc buffer
  size_t packet_size = sizeof(packet_t);
  TCA_CHECK(tcaMalloc((void**)&ring_buf->send_buffer, packet_size * RING_SIZE, tcaMemoryCPU));
  TCA_CHECK(tcaMalloc((void**)&ring_buf->recv_buffer, packet_size * RING_SIZE, tcaMemoryCPU));
  memset(ring_buf->send_buffer, 0, packet_size * RING_SIZE);
  memset(ring_buf->recv_buffer, 0, packet_size * RING_SIZE);
  
  //create buffer handles
  /* for(int i = 0; i < RING_SIZE; i++){ */
  /*   TCA_CHECK(tcaCreateHandle(&(ring_buf->local_buffer_handles[i]), &(ring_buf->buffer[i]), packet_size, tcaMemoryCPU)); */
  /* } */
  TCA_CHECK(tcaCreateHandle(&(ring_buf->local_send_buffer_handle), ring_buf->send_buffer, packet_size * RING_SIZE, tcaMemoryCPU));
  TCA_CHECK(tcaCreateHandle(&(ring_buf->local_recv_buffer_handle), ring_buf->recv_buffer, packet_size * RING_SIZE, tcaMemoryCPU));
  
  //exchange buffer handles
  MPI_Sendrecv(&(ring_buf->local_send_buffer_handle), sizeof(tcaHandle), MPI_BYTE, remote_rank, _handle_sendrecv_tag,
	       &(ring_buf->remote_send_buffer_handle), sizeof(tcaHandle), MPI_BYTE, remote_rank, _handle_sendrecv_tag,
	       MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(&(ring_buf->local_recv_buffer_handle), sizeof(tcaHandle), MPI_BYTE, remote_rank, _handle_sendrecv_tag,
	       &(ring_buf->remote_recv_buffer_handle), sizeof(tcaHandle), MPI_BYTE, remote_rank, _handle_sendrecv_tag,
	       MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  for(int i = 0; i < RING_SIZE; i++){
    tcaDesc* desc = tcaDescNew();
    int buf_wait_slot = _wait_slot_offset + i;
    TCA_CHECK(tcaDescSetMemcpy(desc,
			       &(ring_buf->remote_recv_buffer_handle), packet_size * i,
			       &(ring_buf->local_send_buffer_handle), packet_size * i,
			       packet_size,
			       tcaDMAUseInternal | tcaDMAUseNotifyInternal,
			       //tcaDMAUseInternal | tcaDMAUseNotifyInternal | tcaDMANotifySelf | tcaDMANotify,
			       buf_wait_slot,
			       _packet_wait_tag));
    TCA_CHECK(tcaDescSetMemcpy(desc,
			       &(ring_buf->remote_psn_pairs_handle), psn_size * 2,
			       &(ring_buf->local_psn_pairs_handle), 0,
			       psn_size * 2,
#ifdef _USE_NOTIFY_TO_WAIT_PACKET
			       tcaDMAUseInternal | tcaDMAUseNotifyInternal | tcaDMANotifySelf | tcaDMANotify,
			       //tcaDMAUseInternal | tcaDMAUseNotifyInternal,
#else
			       tcaDMAUseInternal | tcaDMAUseNotifyInternal | tcaDMANotifySelf,
#endif
			       buf_wait_slot,
			       _packet_wait_tag));
    ring_buf->buffer_desc[i] = desc;
  }
}

static void _XMP_tca_ring_buf_finalize(tca_ring_buf_t* ring_buf)
{
  //destroy and free my buffer handles
  /* for(int i = 0; i < RING_SIZE; i++){ */
  /*   TCA_CHECK(tcaDestroyHandle(&ring_buf->local_buffer_handles[i])); */
  /* } */
  TCA_CHECK(tcaDestroyHandle(&ring_buf->local_send_buffer_handle));
  TCA_CHECK(tcaDestroyHandle(&ring_buf->local_recv_buffer_handle));

  //free my buffer
  //_XMP_free(ring_buf->buffer);
  TCA_CHECK(tcaFree(ring_buf->send_buffer, tcaMemoryCPU));
  TCA_CHECK(tcaFree(ring_buf->recv_buffer, tcaMemoryCPU));
  TCA_CHECK(tcaFree(ring_buf->psn_pairs, tcaMemoryCPU));
}

void _XMP_tca_comm_init()
{
  _ring_bufs = _XMP_alloc(sizeof(tca_ring_buf_t) * _XMP_world_size);
  for(int i = 0; i < _XMP_world_size; i++){
    if(i == _XMP_world_rank) continue;
    _XMP_tca_ring_buf_init(&_ring_bufs[i], i);
  }
}
static unsigned long long spin_wait_count=0;
  
void _XMP_tca_comm_finalize()
{
  for(int i = 0; i < _XMP_world_size; i++){
    if(i == _XMP_world_rank) continue;
    _XMP_tca_ring_buf_finalize(&_ring_bufs[i]);
  }
  _XMP_free(_ring_bufs);
  //fprintf(stderr, "spin_wait_count=%llu\n", spin_wait_count);
}

static void _XMP_tca_ring_buf_send(tca_ring_buf_t *ring_buf, const int tag, const int data) //postの場合 tag=post_req, packet=post_tag
{
  volatile psn_t *r_recv_psn = &(ring_buf->psn_pairs[3]);
  const unsigned long long local_send_psn = ring_buf->psn_pairs[0]; //0: local_send

  //fprintf(stderr,"send start (tag=%d, data=%d, psn=%llu)\n", tag, data, local_send_psn);
  _XMP_tca_lock();

  //相手のring bufferが空くまで待機
  while(local_send_psn - *r_recv_psn >= RING_SIZE){
    const int psn_wait_slot = _wait_slot_offset + RING_SIZE;
    TCA_CHECK(tcaWaitDMARecvDesc(&(ring_buf->remote_psn_pairs_handle), psn_wait_slot, _psn_wait_tag));
  }

  const int buf_pos = local_send_psn % RING_SIZE;
  const int buf_wait_slot = _wait_slot_offset + buf_pos;

  //send_bufferに値の書き込み
  ring_buf->send_buffer[buf_pos].tag = tag;
  ring_buf->send_buffer[buf_pos].data = data;
  //update local send psn
  ring_buf->psn_pairs[0] = local_send_psn + 1;
  ring_buf->last_send_local_recv_psn = ring_buf->psn_pairs[1];
  
  // set desc
  TCA_CHECK(tcaDescSet(ring_buf->buffer_desc[buf_pos], _dmac_channel));
  // start dmac
  TCA_CHECK(tcaStartDMADesc(_dmac_channel));

  // wait put locally
  TCA_CHECK(tcaWaitDMARecvDesc(&(ring_buf->local_send_buffer_handle), buf_wait_slot, _packet_wait_tag));

  //wait dmac
  TCA_CHECK(tcaWaitDMAC(_dmac_channel)); //important

  //fprintf(stderr,"send end (tag=%d, data=%d, psn=%llu)\n", tag, data, local_send_psn);

  _XMP_tca_unlock();
}

void _XMP_tca_comm_send(const int rank, const int tag, const int data) //postの場合 tag=post_req, packet=post_tag
{
  if(rank == _XMP_world_rank){
    _XMP_fatal("send to self is not supported");
  }
  _XMP_tca_ring_buf_send(&_ring_bufs[rank], tag, data);
}

static void _XMP_tca_ring_buf_recv(tca_ring_buf_t *ring_buf, int *tag, int *data)
{
  psn_t local_recv_psn = ring_buf->psn_pairs[1]; //1: local_recv
  int buf_pos = local_recv_psn % RING_SIZE;

  _XMP_tca_lock();

#ifdef _USE_NOTIFY_TO_WAIT_PACKET
  //wait notify for recv_buf
  const int buf_wait_slot = _wait_slot_offset + buf_pos;
  TCA_CHECK(tcaWaitDMARecvDesc(&(ring_buf->remote_psn_pairs_handle), buf_wait_slot, _packet_wait_tag));
#else
  //wait until (remote_send_psn > local_recv_psn) become true
  volatile psn_t* remote_send_psn_p = &(ring_buf->psn_pairs[2]); //2: remote_send
  while(*remote_send_psn_p <= local_recv_psn){
    ++spin_wait_count;
    _mm_pause();
  }
#endif

  //copy recved data
  *tag = ring_buf->recv_buffer[buf_pos].tag;
  *data = ring_buf->recv_buffer[buf_pos].data;

  //fprintf(stderr, "recv: psn=%ull, tag=%d, data=%d\n", local_recv_psn, *tag, *data);

  //update local_recv_psn
  ring_buf->psn_pairs[1] = ++local_recv_psn;

  if(local_recv_psn - ring_buf->last_send_local_recv_psn >= (RING_SIZE/2)){
    //send local psn pair
    const int psn_wait_slot = _wait_slot_offset + RING_SIZE;
    ring_buf->last_send_local_recv_psn = ring_buf->psn_pairs[1];
    TCA_CHECK(tcaDescSet(ring_buf->psn_desc, _dmac_channel));
    TCA_CHECK(tcaStartDMADesc(_dmac_channel));
    TCA_CHECK(tcaWaitDMARecvDesc(&(ring_buf->local_psn_pairs_handle), psn_wait_slot, _psn_wait_tag));
    TCA_CHECK(tcaWaitDMAC(_dmac_channel)); //important
  }

  //printf("psns: %llu, %llu, %llu, %llu\n", ring_buf->psn_pairs[0],ring_buf->psn_pairs[1],ring_buf->psn_pairs[2],ring_buf->psn_pairs[3]);
  _XMP_tca_unlock();
}

static bool _XMP_tca_ring_buf_recv_nowait(tca_ring_buf_t *ring_buf, packet_t *packet)
{
  psn_t local_recv_psn = ring_buf->psn_pairs[1]; //1: local_recv

  //wait until (remote_send_psn > local_recv_psn) become true
  volatile psn_t* remote_send_psn_p = &(ring_buf->psn_pairs[2]); //2: remote_send
  if(*remote_send_psn_p <= local_recv_psn){
    return false;
  }    

  //copy recved data
  int buf_pos = local_recv_psn % RING_SIZE;
  *packet = ring_buf->recv_buffer[buf_pos];

  //fprintf(stderr, "recv: psn=%ull, tag=%d, data=%d\n", local_recv_psn, *tag, *data);

  //update local_recv_psn
  ring_buf->psn_pairs[1] = ++local_recv_psn;

  if(local_recv_psn - ring_buf->last_send_local_recv_psn >= (RING_SIZE/2)){
    _XMP_tca_lock();
    //send local psn pair
    const int psn_wait_slot = _wait_slot_offset + RING_SIZE;
    ring_buf->last_send_local_recv_psn = ring_buf->psn_pairs[1];
    TCA_CHECK(tcaDescSet(ring_buf->psn_desc, _dmac_channel));
    TCA_CHECK(tcaStartDMADesc(_dmac_channel));
    TCA_CHECK(tcaWaitDMARecvDesc(&(ring_buf->local_psn_pairs_handle), psn_wait_slot, _psn_wait_tag));
    TCA_CHECK(tcaWaitDMAC(_dmac_channel)); //important
    _XMP_tca_unlock();
  }

  return true;
}

bool _XMP_tca_comm_recv_nowait(const int rank, packet_t *packet){
  if(rank == _XMP_world_rank){
    _XMP_fatal("recv from self is not supported");
    return false;
  }
  return _XMP_tca_ring_buf_recv_nowait(&_ring_bufs[rank], packet);
}

void _XMP_tca_comm_recv(const int rank, int *tag, int *data) //in:rank, out:tag,packet
{
  if(rank == _XMP_world_rank){
    _XMP_fatal("recv from self is not supported");
  }
  _XMP_tca_ring_buf_recv(&_ring_bufs[rank], tag, data);
}


/////////////////
// comm threads
/////////////////

double getElapsedTime_(struct timespec *begin, struct timespec *end){
  return (end->tv_sec - begin->tv_sec) + (end->tv_nsec - begin->tv_nsec) * 1e-9;
}
struct timespec begin_ts, end_ts;

static void* comm_thread_func(void* param)
{
  //printf("start comm thread\n");

  volatile char *flag_p = &_run_flag;

  while(*flag_p){
    for(int rank = 0; rank < _XMP_world_size; ++rank){
      if(rank == _XMP_world_rank) continue;

      packet_t packet;
      bool is_recved = _XMP_tca_comm_recv_nowait(rank, &packet);
      if(! is_recved) continue;

      //      _XMP_tca_comm_recv(rank, &packet.tag, &packet.data);

      //fprintf(stderr, "recved packet (tag:%d, data:%d)\n", packet.tag, packet.data);

      //process recved packet
      //1. post_req
      if(packet.tag == _XMP_TCA_POSTREQ_TAG){
	_xmp_tca_postreq(rank, packet.data);
	//if(_XMP_world_rank==1) fprintf(stderr, "postreq time=%f\n", getElapsedTime_(&begin_ts, &end_ts)*1000*1000);
	//_mm_sfence();
      }
      
      //2. get_req
      //3. ???
    }
  }

  return NULL;
}

static void create_comm_thread()
{
  //printf("create comm thread\n");
  if(_run_flag != 0){
    _XMP_fatal("comm thread is already running");
  }

  _run_flag = 1;
  if(pthread_create(&comm_thread, NULL/*&attr*/, comm_thread_func, NULL)){
    _XMP_fatal("failed to create comm thread\n");
  }
}

static void destroy_comm_thread()
{
  //printf("destroy comm thread\n");
  if(_run_flag == 0){
    return;
  }
  
  _run_flag = 0;
  pthread_join(comm_thread, NULL);
}

void _XMP_tca_lock()
{
  int ret = pthread_mutex_lock(&_XMP_tca_mutex);
  if(ret == 0) return;

  _XMP_fatal("tca lock failed");
}

void _XMP_tca_unlock()
{
  int ret = pthread_mutex_unlock(&_XMP_tca_mutex);
  if(ret == 0) return;
  _XMP_fatal("tca unlock failed");
}
