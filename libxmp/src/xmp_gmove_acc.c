#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "xmp.h"
#include "xmp_internal.h"
#include "xmp_math_function.h"

#define _XACC_NUM_COMM_CACHES 4
#define _XACC_MAX_NUM_SENDRECVS 4

typedef struct _XACC_sendrecv_comm_type{
  void *buf;
  int count;
  MPI_Datatype datatype;
  int rank;
  MPI_Comm comm;
}_XACC_sendrecv_comm_t;

typedef struct _XACC_gmv_comm_type{
  _XMP_gmv_desc_t *desc_left;
  _XMP_gmv_desc_t *desc_right;
  int num_recvs;
  int num_sends;
  /*
  MPI_Request *req_recvs;
  MPI_Request *req_sends;
  */
  _XACC_sendrecv_comm_t recvs[_XACC_MAX_NUM_SENDRECVS];
  _XACC_sendrecv_comm_t sends[_XACC_MAX_NUM_SENDRECVS];
  /*
  void *send_buf;
  int send_count;
  MPI_Datatype send_datatype;
  int dst;
  MPI_Comm send_comm;

  void *recv_buf;
  int recv_count;
  MPI_Datatype recv_datatype;
  int src;
  MPI_Comm recv_comm;
  */
}_XACC_gmv_comm_t;


static _XACC_gmv_comm_t *g_comm_cache[_XACC_NUM_COMM_CACHES];
static int g_comm_cache_tail = 0;
static _XACC_gmv_comm_t g_current_comm;

static _XACC_gmv_comm_t* find_gmv_comm(_XMP_gmv_desc_t *gmv_desc_leftp, _XMP_gmv_desc_t *gmv_desc_rightp);
static void exec_gmv_comm(_XACC_gmv_comm_t * comm);
static _XMP_gmv_desc_t* alloc_gmv_desc(_XMP_gmv_desc_t *desc);
static void cache_comm(_XACC_gmv_comm_t *comm);
static void set_sendrecv_comm(_XACC_sendrecv_comm_t *sendrecv_comm, void *buf, int count, MPI_Datatype datatype, int rank, MPI_Comm comm);

//decl
static void array_array_common(_XMP_gmv_desc_t *gmv_desc_leftp, _XMP_gmv_desc_t *gmv_desc_rightp,
				 int *dst_l, int *dst_u, int *dst_s, unsigned long long *dst_d,
				 int *src_l, int *src_u, int *src_s, unsigned long long *src_d);
static void sendrecv_ARRAY(int type, int type_size, /*MPI_Datatype *mpi_datatype,*/
			   _XMP_array_t *dst_array, void *dst_array_dev, int *dst_array_nodes_ref,
			   int *dst_lower, int *dst_upper, int *dst_stride, unsigned long long *dst_dim_acc,
			   _XMP_array_t *src_array, void *src_array_dev, int *src_array_nodes_ref,
			     int *src_lower, int *src_upper, int *src_stride, unsigned long long *src_dim_acc);


void _XMP_gmove_acc_LOCALCOPY_ARRAY(int type, size_t type_size, ...)
{
  _XMP_fatal("_XMP_gmove_acc_LOCALCOPY_ARRAY is not implemented!\n");
}
void _XMP_gmove_acc_BCAST_ARRAY(_XMP_array_t *src_array, int type, size_t type_size, ...)
{
  _XMP_fatal("_XMP_gmove_acc_BCAST_ARRAY is not implemented!\n");
}
void _XMP_gmove_acc_HOMECOPY_ARRAY(_XMP_array_t *dst_array, int type, size_t type_size, ...)
{
  _XMP_fatal("_XMP_gmove_acc_HOMECOPY_ARRAY is not implemented!\n");
}
void _XMP_gmove_acc_BCAST_TO_NOTALIGNED_ARRAY(_XMP_array_t *dst_array, _XMP_array_t *src_array, int type, size_t type_size, ...)
{
  _XMP_fatal("_XMP_gmove_acc_BCAST_TO_NOTALIGNED_ARRAY is not implemented!\n");
}
void _XMP_gmove_acc_bcast_SCALAR(void *dst_addr, void *src_addr, size_t type_size, int root_rank)
{
  _XMP_fatal("_XMP_gmove_acc_bcast_SCALAR is not implemented!\n");
}

int _XMP_gmove_acc_HOMECOPY_SCALAR(_XMP_array_t *array, ...)
{
  _XMP_fatal("_XMP_gmove_acc_bcast_HOMECOPY_SCALAR is not implemented!\n");
  return 0; //dummy
}
void _XMP_gmove_acc_SENDRECV_SCALAR(void *dst_addr, void *src_addr,
                                _XMP_array_t *dst_array, _XMP_array_t *src_array, ...)
{
  _XMP_fatal("_XMP_gmove_acc_bcast_SENDRECV_SCALAR is not implemented!\n");
}


static _Bool is_same_axis(_XMP_array_t *adesc0, _XMP_array_t *adesc1)
{
  if (adesc0->dim != adesc1->dim) return false;

  for (int i = 0; i < adesc0->dim; i++) {
    if (adesc0->info[i].align_template_index 
        != adesc1->info[i].align_template_index) return false;
  }

  return true;
}

static _Bool is_same_offset(_XMP_array_t *adesc0, _XMP_array_t *adesc1)
{
  if (adesc0->dim != adesc1->dim) return false;

  for (int i = 0; i < adesc0->dim; i++) {
    if (adesc0->info[i].align_subscript
        != adesc1->info[i].align_subscript) return false;
  }

  return true;
}

static _Bool is_same_alignmanner(_XMP_array_t *adesc0, _XMP_array_t *adesc1)
{
  _XMP_template_t *t0 = (_XMP_template_t *)adesc0->align_template;
  _XMP_template_t *t1 = (_XMP_template_t *)adesc1->align_template;
  int taxis0, taxis1, naxis0, naxis1, nsize0, nsize1;

  if (adesc0->dim != adesc1->dim) return false;

  for (int i = 0; i < adesc0->dim; i++) {
    int idim0 = adesc0->info[i].align_template_index;
    int idim1 = adesc1->info[i].align_template_index;
    if ((adesc0->info[i].align_manner == _XMP_N_ALIGN_DUPLICATION)
        || (adesc1->info[i].align_manner == _XMP_N_ALIGN_DUPLICATION)){
       return false;
    }else if (adesc0->info[i].align_manner != adesc1->info[i].align_manner){
       return false;
    }else{
       if(adesc0->info[i].align_manner==_XMP_N_ALIGN_BLOCK_CYCLIC
         && t0->chunk[idim0].par_width != t1->chunk[idim1].par_width){
         return false;
       }else if(adesc0->info[i].align_manner==_XMP_N_ALIGN_GBLOCK){
         xmp_align_axis(adesc0, i+1, &taxis0);
         xmp_align_axis(adesc1, i+1, &taxis1);
         xmp_dist_axis(t0, taxis0, &naxis0);
         xmp_dist_axis(t1, taxis1, &naxis1);
         xmp_nodes_size(adesc0->array_nodes, naxis0, &nsize0);
         xmp_nodes_size(adesc1->array_nodes, naxis1, &nsize1);
         int map0[nsize0], map1[nsize1];
         xmp_dist_gblockmap(t0, naxis0, map0);
         xmp_dist_gblockmap(t1, naxis1, map1);
         if (nsize0 == nsize1){
           for(int ii=0; ii<nsize0; ii++){
             if (map0[ii] != map1[ii]){
                return false;
             }
           }
         }else{
           return false;
         }
       }
    }
  }

  return true;

}

static _Bool is_same_array_shape(_XMP_array_t *adesc0, _XMP_array_t *adesc1)
{
  if (adesc0->dim != adesc1->dim) return false;

  for (int i = 0; i < adesc0->dim; i++) {
    if (adesc0->info[i].ser_lower != adesc1->info[i].ser_lower ||
	adesc0->info[i].ser_upper != adesc1->info[i].ser_upper) return false;
  }

  return true;
}

static _Bool is_same_template_shape(_XMP_template_t *tdesc0, _XMP_template_t *tdesc1)
{
  if (tdesc0->dim != tdesc1->dim) return false;

  for (int i = 0; i < tdesc0->dim; i++) {
    if (tdesc0->info[i].ser_lower != tdesc1->info[i].ser_lower ||
        tdesc0->info[i].ser_upper != tdesc1->info[i].ser_upper) return false;
  }

  return true;
}

static _Bool is_whole(_XMP_gmv_desc_t *gmv_desc)
{
  _XMP_array_t *adesc = gmv_desc->a_desc;

  for (int i = 0; i < adesc->dim; i++){
    if (gmv_desc->lb[i] == 0 && gmv_desc->ub[i] == 0 && gmv_desc->st[i] == 0) continue;
    if (adesc->info[i].ser_lower != gmv_desc->lb[i] ||
	adesc->info[i].ser_upper != gmv_desc->ub[i] ||
	gmv_desc->st[i] != 1) return false;
  }

  return true;
}

static _Bool is_one_block(_XMP_array_t *adesc)
{
  int cnt = 0;

  for (int i = 0; i < adesc->dim; i++) {
    if (adesc->info[i].align_manner == _XMP_N_ALIGN_BLOCK) cnt++;
    else if (adesc->info[i].align_manner != _XMP_N_ALIGN_NOT_ALIGNED) return false;
  }
  
  if (cnt != 1) return false;
  else return true;
}


void _XMP_gmove_acc_SENDRECV_ARRAY(_XMP_array_t *dst_array, _XMP_array_t *src_array,
				   void *dst_array_dev, void *src_array_dev,
                               int type, size_t type_size, ...)
{

  _XMP_gmv_desc_t gmv_desc_leftp, gmv_desc_rightp;
  //unsigned long long gmove_total_elmts = 0;

  va_list args;
  va_start(args, type_size);

  // get dst info
  unsigned long long dst_total_elmts = 1;
  //void *dst_addr = dst_array->array_addr_p;
  int dst_dim = dst_array->dim;
  int dst_l[dst_dim], dst_u[dst_dim], dst_s[dst_dim]; unsigned long long dst_d[dst_dim];
  for (int i = 0; i < dst_dim; i++) {
    dst_l[i] = va_arg(args, int);
    dst_u[i] = dst_l[i] + va_arg(args, int)-1;
    dst_s[i] = va_arg(args, int);
    dst_d[i] = va_arg(args, unsigned long long);
    _XMP_normalize_array_section(&gmv_desc_leftp, i, &(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
    dst_total_elmts *= _XMP_M_COUNT_TRIPLETi(dst_l[i], dst_u[i], dst_s[i]);
  }

  // get src info
  unsigned long long src_total_elmts = 1;
  //void *src_addr = src_array->array_addr_p;
  int src_dim = src_array->dim;;
  int src_l[src_dim], src_u[src_dim], src_s[src_dim]; unsigned long long src_d[src_dim];
  for (int i = 0; i < src_dim; i++) {
    src_l[i] = va_arg(args, int);
    src_u[i] = src_l[i] + va_arg(args, int)-1;
    src_s[i] = va_arg(args, int);
    src_d[i] = va_arg(args, unsigned long long);
    _XMP_normalize_array_section(&gmv_desc_rightp, i, &(src_l[i]), &(src_u[i]), &(src_s[i]));
    src_total_elmts *= _XMP_M_COUNT_TRIPLETi(src_l[i], src_u[i], src_s[i]);
  }
  va_end(args);

  if (dst_total_elmts != src_total_elmts) {
    _XMP_fatal("bad assign statement for gmove");
  } else {
    //gmove_total_elmts = dst_total_elmts;
  }

  // do transpose
  int dummy[7] = { 2, 2, 2, 2, 2, 2, 2 }; /* temporarily assuming maximum 7-dimensional */

  gmv_desc_leftp.is_global = true;       gmv_desc_rightp.is_global = true;
  gmv_desc_leftp.ndims = dst_array->dim; gmv_desc_rightp.ndims = src_array->dim;

  gmv_desc_leftp.a_desc = dst_array;     gmv_desc_rightp.a_desc = src_array;

  gmv_desc_leftp.local_data = NULL;      gmv_desc_rightp.local_data = NULL;
  gmv_desc_leftp.a_lb = NULL;            gmv_desc_rightp.a_lb = NULL;
  gmv_desc_leftp.a_ub = NULL;            gmv_desc_rightp.a_ub = NULL;

  gmv_desc_leftp.kind = dummy;           gmv_desc_rightp.kind = dummy; // always triplet
  gmv_desc_leftp.lb = dst_l;             gmv_desc_rightp.lb = src_l;
  gmv_desc_leftp.ub = dst_u;             gmv_desc_rightp.ub = src_u;
  gmv_desc_leftp.st = dst_s;             gmv_desc_rightp.st = src_s;

  gmv_desc_leftp.a_dev = dst_array_dev;  gmv_desc_rightp.a_dev = src_array_dev;

  _XACC_gmv_comm_t *gmv_comm = find_gmv_comm(&gmv_desc_leftp, &gmv_desc_rightp);
  if(gmv_comm == NULL){
    //    printf("not found gmv_comm\n");
    g_current_comm.num_recvs = 0;
    g_current_comm.num_sends = 0;

    array_array_common(&gmv_desc_leftp, &gmv_desc_rightp, dst_l, dst_u, dst_s, dst_d, src_l, src_u, src_s, src_d);

    if(g_current_comm.num_recvs <= _XACC_MAX_NUM_SENDRECVS && g_current_comm.num_sends <= _XACC_MAX_NUM_SENDRECVS){
      gmv_comm = (_XACC_gmv_comm_t*)malloc(sizeof(_XACC_gmv_comm_t));
      gmv_comm->desc_left = alloc_gmv_desc(&gmv_desc_leftp);
      gmv_comm->desc_right = alloc_gmv_desc(&gmv_desc_rightp);
      gmv_comm->num_recvs = g_current_comm.num_recvs;
      gmv_comm->num_sends = g_current_comm.num_sends;

      for(int i = 0; i < g_current_comm.num_recvs; i++){
	gmv_comm->recvs[i] = g_current_comm.recvs[i];
      }
      for(int i = 0; i < g_current_comm.num_sends; i++){
	gmv_comm->sends[i] = g_current_comm.sends[i];
      }
      cache_comm(gmv_comm);
    }
  }else{
    //do cached comm
    exec_gmv_comm(gmv_comm);
  }
}

static void array_array_common(_XMP_gmv_desc_t *gmv_desc_leftp, _XMP_gmv_desc_t *gmv_desc_rightp, int *dst_l, int *dst_u, int *dst_s, unsigned long long  *dst_d, int *src_l, int *src_u, int *src_s, unsigned long long *src_d){

  _XMP_array_t *dst_array=NULL;
  _XMP_array_t *src_array = gmv_desc_rightp->a_desc;
  _XMP_template_t *dst_template=NULL, *src_template=NULL;
  _XMP_nodes_t *dst_nodes=NULL, *src_nodes=NULL;

  void *dst_addr=NULL, *src_addr=NULL;
  int type=-1;
  size_t type_size=0;
  int dst_dim = gmv_desc_leftp->ndims;
  int src_dim = gmv_desc_rightp->ndims;
  int dst_num=-1, src_num=-1, dst_sub_dim=-1, src_sub_dim=-1;
  int dst_comm_size=-1, src_comm_size=-1;
  int exec_comm_size = _XMP_get_execution_nodes()->comm_size;

  if (gmv_desc_leftp->is_global == true && gmv_desc_rightp->is_global == true){
    dst_array = gmv_desc_leftp->a_desc;
    type = dst_array->type;
    type_size = dst_array->type_size;
    dst_addr = dst_array->array_addr_p;
    src_addr = src_array->array_addr_p;
    dst_template = dst_array->align_template;
    src_template = src_array->align_template;
    dst_nodes = dst_template->onto_nodes;
    src_nodes = src_template->onto_nodes;
    dst_comm_size = dst_nodes->comm_size;
    src_comm_size = src_nodes->comm_size;
  }else if(gmv_desc_leftp->is_global == false && gmv_desc_rightp->is_global == true){
    dst_addr = gmv_desc_leftp->local_data;
    src_addr = src_array->array_addr_p;
    type = src_array->type;
    type_size = src_array->type_size;
  }else if(gmv_desc_leftp->is_global == true && gmv_desc_rightp->is_global == false){
    dst_array = gmv_desc_leftp->a_desc;
    type = dst_array->type;
    type_size = dst_array->type_size;
    dst_addr = dst_array->array_addr_p;
    src_addr = gmv_desc_rightp->local_data;
  }

  for (int i=0;i<dst_dim;i++){
    dst_num=_XMP_M_COUNT_TRIPLETi(dst_l[i], dst_u[i], dst_s[i]);
    if (dst_num > 1) {
      dst_sub_dim++;
    }
  }

  for (int i=0;i<src_dim;i++){
    src_num=_XMP_M_COUNT_TRIPLETi(src_l[i], src_u[i], src_s[i]);
    if (src_num > 1) {
      src_sub_dim++;
    }
  }

  if ((gmv_desc_leftp->is_global == true) && (gmv_desc_rightp->is_global == true)){
    if ((exec_comm_size == dst_comm_size) && (exec_comm_size == src_comm_size)){
        if (is_same_array_shape(dst_array, src_array) &&
           is_same_template_shape(dst_array->align_template, src_array->align_template) &&
           is_same_axis(dst_array, src_array) &&
           is_same_offset(dst_array, src_array) &&
           is_same_alignmanner(dst_array, src_array) &&
           is_whole(gmv_desc_leftp) && is_whole(gmv_desc_rightp)) {


        for (int i = 0; i < dst_dim; i++) {
          dst_l[i]=dst_array->info[i].local_lower;
          dst_u[i]=dst_array->info[i].local_upper;
          dst_s[i]=dst_array->info[i].local_stride;
        }

        for (int i = 0; i < src_dim; i++) {
          src_l[i]=src_array->info[i].local_lower;
          src_u[i]=src_array->info[i].local_upper;
          src_s[i]=src_array->info[i].local_stride;
        }

        _XMP_gmove_localcopy_ARRAY(type, type_size,
                                   (void *)dst_addr, dst_dim, dst_l, dst_u, dst_s, dst_d,
                                   (void *)src_addr, src_dim, src_l, src_u, src_s, src_d);
        return;
      }

      if (is_same_array_shape(dst_array, src_array) &&
          is_whole(gmv_desc_leftp) && is_whole(gmv_desc_rightp) &&
          is_one_block(dst_array) && is_one_block(src_array) &&
          (dst_array->dim >= dst_array->align_template->dim) &&
          (src_array->dim >= src_array->align_template->dim)){
	_XMP_fatal("_XMP_gmove_transpose is umimplemented\n");
        //if (_XMP_gmove_transpose(gmv_desc_leftp, gmv_desc_rightp)) return;
      }
    }
  }

// temporary check flag : chk_flag

  int chk_flag=0, dst_chk_flag, src_chk_flag;

  if (gmv_desc_leftp->is_global == true
     && gmv_desc_rightp->is_global == true){

     for(int i=0; i<dst_template->dim;i++){
       if ((dst_template->chunk[i].dist_manner !=_XMP_N_DIST_BLOCK)
          && (dst_template->chunk[i].dist_manner !=_XMP_N_DIST_CYCLIC)){
          dst_chk_flag=0;
          break;
       }else{
          dst_chk_flag=1;
       }
     }

     for(int i=0; i<src_template->dim;i++){
       if ((src_template->chunk[i].dist_manner !=_XMP_N_DIST_BLOCK)
          && (src_template->chunk[i].dist_manner !=_XMP_N_DIST_CYCLIC)){
          src_chk_flag=0;
          break;
       }else{
          src_chk_flag=1;
       }
     }

     if (dst_dim==dst_nodes->dim){
        if(src_dim==src_nodes->dim){

          for(int i=0; i<dst_dim;i++){
            if ((dst_array->info[i].align_subscript != 0 )
               || ((dst_array->info[i].align_manner !=_XMP_N_ALIGN_BLOCK)
                  && (dst_array->info[i].align_manner !=_XMP_N_ALIGN_CYCLIC))){
              dst_chk_flag=0;
              break;
            }
          }

          for(int i=0; i<src_dim;i++){
            if ((src_array->info[i].align_subscript != 0 )
               || ((src_array->info[i].align_manner !=_XMP_N_ALIGN_BLOCK)
                  && (src_array->info[i].align_manner !=_XMP_N_ALIGN_CYCLIC))){
                src_chk_flag=0;
                break;
             }
          }

        }

        if (is_whole(gmv_desc_leftp) && is_whole(gmv_desc_rightp)){
        }else{
          dst_chk_flag=0;
          src_chk_flag=0;
        }

     }else if (dst_dim < dst_nodes->dim){
        if(src_dim < src_nodes->dim){
          if((dst_nodes->dim != 2) 
            || (src_nodes->dim != 2)
            || (dst_array->info[0].align_subscript != 0)
            || (src_array->info[0].align_subscript != 0)){
            dst_chk_flag=0;
            src_chk_flag=0;
          }
        }else{
          dst_chk_flag=0;
          src_chk_flag=0;
        }
     }else if (dst_dim > dst_nodes->dim){
        if (src_dim > src_nodes->dim){
           if (_XMPF_running == 1
               && _XMPC_running == 0){
              dst_chk_flag=0;
              src_chk_flag=0;
           }
        }else{
           dst_chk_flag=0;
           src_chk_flag=0;
        }
     }

     if (dst_chk_flag==1 && src_chk_flag==1) chk_flag=1;

     if ((exec_comm_size != dst_comm_size) || (exec_comm_size != src_comm_size)) chk_flag=0;

  }

  if (chk_flag == 1) {
    /*
    MPI_Datatype mpi_datatype;
    MPI_Type_contiguous(type_size, MPI_BYTE, &mpi_datatype);
    MPI_Type_commit(&mpi_datatype);
    */
    _XMP_nodes_t *dst_array_nodes = dst_array->array_nodes;
    int dst_array_nodes_dim = dst_array_nodes->dim;
    int dst_array_nodes_ref[dst_array_nodes_dim];
    for (int i = 0; i < dst_array_nodes_dim; i++) {
      dst_array_nodes_ref[i] = 0;
    }


    for(int i=0;i<dst_dim;i++){
      if (dst_array->info[i].align_manner ==_XMP_N_ALIGN_BLOCK_CYCLIC){
        break;
      }
    }

    for(int i=0;i<src_dim;i++){
      if (src_array->info[i].align_manner ==_XMP_N_ALIGN_BLOCK_CYCLIC){
        break;
      }
    }

    _XMP_nodes_t *src_array_nodes = src_array->array_nodes;
    int src_array_nodes_dim = src_array_nodes->dim;
    int src_array_nodes_ref[src_array_nodes_dim];

    int dst_lower[dst_dim], dst_upper[dst_dim], dst_stride[dst_dim];
    int src_lower[src_dim], src_upper[src_dim], src_stride[src_dim];
    do {
      for (int i = 0; i < dst_dim; i++) {
        dst_lower[i] = dst_l[i]; dst_upper[i] = dst_u[i]; dst_stride[i] = dst_s[i];
      }

      for (int i = 0; i < src_dim; i++) {
        src_lower[i] = src_l[i]; src_upper[i] = src_u[i]; src_stride[i] = src_s[i];
      }

      if (_XMP_calc_global_index_BCAST(src_dim, src_lower, src_upper, src_stride,
                                     dst_array, dst_array_nodes_ref, dst_lower, dst_upper, dst_stride)) {
        for (int i = 0; i < src_array_nodes_dim; i++) {
          src_array_nodes_ref[i] = 0;
        }

        int recv_lower[dst_dim], recv_upper[dst_dim], recv_stride[dst_dim];
        int send_lower[src_dim], send_upper[src_dim], send_stride[src_dim];
        do {
          for (int i = 0; i < dst_dim; i++) {
            recv_lower[i] = dst_lower[i]; recv_upper[i] = dst_upper[i]; recv_stride[i] = dst_stride[i];
          }

          for (int i = 0; i < src_dim; i++) {
            send_lower[i] = src_lower[i]; send_upper[i] = src_upper[i]; send_stride[i] = src_stride[i];
          }

          if (_XMP_calc_global_index_BCAST(dst_dim, recv_lower, recv_upper, recv_stride,
                                         src_array, src_array_nodes_ref, send_lower, send_upper, send_stride)) {
            sendrecv_ARRAY(type, type_size, /*&mpi_datatype,*/
			   dst_array, gmv_desc_leftp->a_dev, dst_array_nodes_ref,
			   recv_lower, recv_upper, recv_stride, dst_d,
			   src_array, gmv_desc_rightp->a_dev, src_array_nodes_ref,
			   send_lower, send_upper, send_stride, src_d);
          }
        } while (_XMP_get_next_rank(src_array_nodes, src_array_nodes_ref));
      }
    } while (_XMP_get_next_rank(dst_array_nodes, dst_array_nodes_ref));

    //MPI_Type_free(&mpi_datatype);

  }else {

    _XMP_fatal("gmove_acc: unimplemented pattern\n");
    //    _XMP_gmove_1to1(gmv_desc_leftp, gmv_desc_rightp, dst_l, dst_u, dst_s, dst_d, src_l, src_u, src_s, src_d);
    
  }
}

static int is_contiguous(int dim, int *lower, int *upper, int *stride, unsigned long long *dim_acc)
{
  if (dim == 1 && stride[0] == 1){
    return 1;
  }

  return 0;
}

static void sendrecv_ARRAY(int type, int type_size, /*MPI_Datatype *mpi_datatype,*/
			   _XMP_array_t *dst_array, void *dst_array_dev, int *dst_array_nodes_ref,
			   int *dst_lower, int *dst_upper, int *dst_stride, unsigned long long *dst_dim_acc,
			   _XMP_array_t *src_array, void *src_array_dev, int *src_array_nodes_ref,
			   int *src_lower, int *src_upper, int *src_stride, unsigned long long *src_dim_acc) {
  _XMP_nodes_t *dst_array_nodes = dst_array->array_nodes;
  _XMP_nodes_t *src_array_nodes = src_array->array_nodes;
  void *dst_addr = dst_array_dev; //dst_array->array_addr_p;
  void *src_addr = src_array_dev; //src_array->array_addr_p;
  int dst_dim = dst_array->dim;
  int src_dim = src_array->dim;

  _XMP_nodes_t *exec_nodes = _XMP_get_execution_nodes();
  _XMP_ASSERT(exec_nodes->is_member);

  int exec_rank = exec_nodes->comm_rank;
  MPI_Comm *exec_comm = exec_nodes->comm;

  // calc dst_ranks
  _XMP_nodes_ref_t *dst_ref = _XMP_create_nodes_ref_for_target_nodes(dst_array_nodes, dst_array_nodes_ref, exec_nodes);
  int dst_shrink_nodes_size = dst_ref->shrink_nodes_size;
  int dst_ranks[dst_shrink_nodes_size];
  if (dst_shrink_nodes_size == 1) {
    dst_ranks[0] = _XMP_calc_linear_rank(dst_ref->nodes, dst_ref->ref);
  } else {
    _XMP_translate_nodes_rank_array_to_ranks(dst_ref->nodes, dst_ranks, dst_ref->ref, dst_shrink_nodes_size);
  }

  unsigned long long total_elmts = 1;
  for (int i = 0; i < dst_dim; i++) {
    total_elmts *= _XMP_M_COUNT_TRIPLETi(dst_lower[i], dst_upper[i], dst_stride[i]);
  }

  // calc src_ranks
  _XMP_nodes_ref_t *src_ref = _XMP_create_nodes_ref_for_target_nodes(src_array_nodes, src_array_nodes_ref, exec_nodes);
  int src_shrink_nodes_size = src_ref->shrink_nodes_size;
  int src_ranks[src_shrink_nodes_size];
  if (src_shrink_nodes_size == 1) {
    src_ranks[0] = _XMP_calc_linear_rank(src_ref->nodes, src_ref->ref);
  } else {
    _XMP_translate_nodes_rank_array_to_ranks(src_ref->nodes, src_ranks, src_ref->ref, src_shrink_nodes_size);
  }

  unsigned long long src_total_elmts = 1;
  for (int i = 0; i < src_dim; i++) {
    src_total_elmts *= _XMP_M_COUNT_TRIPLETi(src_lower[i], src_upper[i], src_stride[i]);
  }

  if (total_elmts != src_total_elmts) {
    _XMP_fatal("bad assign statement for gmove");
  }

  int even_send_flag = 0;
  int even_send_region_local_nums = -1;
  int even_send_region_local_idx = -1;
  if(src_dim == 1 && dst_array_nodes->comm_size % src_array_nodes->comm_size == 0){
    int src_size0 = src_upper[0] - src_lower[0] + 1;
    int even_send_region_idx = src_lower[0] / src_size0;
    even_send_region_local_nums = src_array->info->par_size / src_size0;
    even_send_region_local_idx = even_send_region_idx % even_send_region_local_nums;
    even_send_flag =1;
    //printf("p(%d) region_idx = %d, num_local_regions=%d, num_local_regions = %d, localidx = %d\n", exec_rank,even_send_region_idx, even_send_region_size, even_send_region_local_nums, even_send_region_local_idx);
  }

  // recv phase
  void *recv_buffer = NULL;
  void *recv_alloc = NULL;
  int wait_recv = _XMP_N_INT_FALSE;
  MPI_Request gmove_request;
  int is_dst_contiguous = 0;
  for (int i = 0; i < dst_shrink_nodes_size; i++) {
    if (dst_ranks[i] == exec_rank) {
      wait_recv = _XMP_N_INT_TRUE;

      int src_rank;
      if(even_send_flag){
	int src_i = even_send_region_local_idx + even_send_region_local_nums * i;
	src_rank = src_ranks[src_i];
	//	printf("p(%d): src_rank[%d]=%d\n", exec_rank, src_i, src_rank);
      }else{
      if ((dst_shrink_nodes_size == src_shrink_nodes_size) ||
          (dst_shrink_nodes_size <  src_shrink_nodes_size)) {
        src_rank = src_ranks[i];
      } else {
        src_rank = src_ranks[i % src_shrink_nodes_size];
      }
      }

      for (int i = 0; i < dst_dim; i++) {
	_XMP_gtol_array_ref_triplet(dst_array, i, &(dst_lower[i]), &(dst_upper[i]), &(dst_stride[i]));
      }
      is_dst_contiguous = is_contiguous(dst_dim, dst_lower, dst_upper, dst_stride, dst_dim_acc);
      if(is_dst_contiguous){
	recv_buffer = (char*)dst_addr + type_size * dst_lower[0];
      }else{
	recv_alloc = _XMP_alloc(total_elmts * type_size);
	recv_buffer = recv_alloc;
      }
      MPI_Irecv(recv_buffer, total_elmts * type_size, MPI_BYTE, src_rank, _XMP_N_MPI_TAG_GMOVE, *exec_comm, &gmove_request);
      //      fprintf(stderr, "DEBUG: Proc(%d), Irecv(src=%d, total_elmnt=%llu)\n", exec_rank, src_rank, total_elmts);
      //save comm start
      if(g_current_comm.num_recvs < _XACC_MAX_NUM_SENDRECVS){
	set_sendrecv_comm(&(g_current_comm.recvs[g_current_comm.num_recvs]),
			  recv_buffer, total_elmts * type_size, MPI_BYTE, src_rank, *exec_comm);
      }
      g_current_comm.num_recvs++;
      //save comm end
    }
  }

  // send phase
  for (int i = 0; i < src_shrink_nodes_size; i++) {
    if (src_ranks[i] == exec_rank) {
      void *send_buffer = NULL;
      void *send_alloc = NULL;
      for (int j = 0; j < src_dim; j++) {
        _XMP_gtol_array_ref_triplet(src_array, j, &(src_lower[j]), &(src_upper[j]), &(src_stride[j]));
      }
      int is_src_contiguous = is_contiguous(src_dim, src_lower, src_upper, src_stride, src_dim_acc);
      if(is_src_contiguous){
	send_buffer = (char*)src_addr + type_size * src_lower[0];
      }else{
	send_alloc = _XMP_alloc(total_elmts * type_size);
	send_buffer = send_alloc;
	_XMP_pack_array(send_buffer, src_addr, type, type_size, src_dim, src_lower, src_upper, src_stride, src_dim_acc);
      }
      if ((dst_shrink_nodes_size == src_shrink_nodes_size) ||
          (dst_shrink_nodes_size <  src_shrink_nodes_size)) {
	if(even_send_flag){
	  int dst_i = i / even_send_region_local_nums;
	  if(i % even_send_region_local_nums == even_send_region_local_idx && dst_i < dst_shrink_nodes_size){
	    //	  fprintf(stderr, "DEBUG: Proc(%d), Send(dst=%d, total_elmnt=%llu)\n", exec_rank, dst_ranks[dst_i], total_elmts);
	  MPI_Send(send_buffer, total_elmts * type_size, MPI_BYTE, dst_ranks[dst_i], _XMP_N_MPI_TAG_GMOVE, *exec_comm);	  
	  //save comm start
	  if(g_current_comm.num_sends < _XACC_MAX_NUM_SENDRECVS){
	    set_sendrecv_comm(&(g_current_comm.sends[g_current_comm.num_sends]),
			      send_buffer, total_elmts * type_size, MPI_BYTE, dst_ranks[dst_i], *exec_comm);
	  }
	  g_current_comm.num_sends++;
	  //save comm end
	  }
	}else{
        if (i < dst_shrink_nodes_size) {
          MPI_Send(send_buffer, total_elmts * type_size, MPI_BYTE, dst_ranks[i], _XMP_N_MPI_TAG_GMOVE, *exec_comm);
	  //	  fprintf(stderr, "DEBUG: Proc(%d), Send(dst=%d, total_elmnt=%llu)\n", exec_rank, dst_ranks[i], total_elmts);
	  //save comm start
	  if(g_current_comm.num_sends < _XACC_MAX_NUM_SENDRECVS){
	    set_sendrecv_comm(&(g_current_comm.sends[g_current_comm.num_sends]),
			      send_buffer, total_elmts * type_size, MPI_BYTE, dst_ranks[i], *exec_comm);
	  }
	  g_current_comm.num_sends++;
	  //save comm end
        }
	}
      } else {
        int request_size = _XMP_M_COUNT_TRIPLETi(i, dst_shrink_nodes_size - 1, src_shrink_nodes_size);
        MPI_Request *requests = _XMP_alloc(sizeof(MPI_Request) * request_size);

        int request_count = 0;
        for (int j = i; j < dst_shrink_nodes_size; j += src_shrink_nodes_size) {
          MPI_Isend(send_buffer, total_elmts*type_size, MPI_BYTE, dst_ranks[j], _XMP_N_MPI_TAG_GMOVE, *exec_comm,
                    requests + request_count);
	  //	  fprintf(stderr, "DEBUG: Proc(%d), Isend(dst=%d, total_elmnt=%llu)\n", exec_rank, dst_ranks[j], total_elmts);
          request_count++;
        }

        MPI_Waitall(request_size, requests, MPI_STATUSES_IGNORE);
        _XMP_free(requests);
      }

      _XMP_free(send_alloc);
    }
  }

  // wait recv phase
  if (wait_recv) {
    MPI_Wait(&gmove_request, MPI_STATUS_IGNORE);
    if(! is_dst_contiguous){
      _XMP_unpack_array(dst_addr, recv_buffer, type, type_size, dst_dim, dst_lower, dst_upper, dst_stride, dst_dim_acc);
    }
    _XMP_free(recv_alloc);
  }

  _XMP_free(dst_ref);
  _XMP_free(src_ref);
}


static int is_same_desc(_XMP_gmv_desc_t *l, _XMP_gmv_desc_t *r)
{
  if ( l->is_global != r->is_global ||
       l->ndims != r->ndims ||
       l->a_desc != r->a_desc || 
       l->local_data != r->local_data ||
       l->a_dev != r->a_dev){
    return _XMP_N_INT_FALSE;
  }
  

  int ndims = l->ndims;
  if(l->local_data != NULL){
    for(int i = 0; i < ndims; i++){
      if(l->a_lb[i] != r->a_lb[i] ||
	 l->a_ub[i] != r->a_ub[i]){
	return _XMP_N_INT_FALSE;
      }
    }
  }

  if(l->a_desc != NULL){
    for(int i = 0; i < ndims; i++){
      if(l->lb[i] != r->lb[i] ||
	 l->ub[i] != r->ub[i] ||
	 l->st[i] != r->st[i]){
	return _XMP_N_INT_FALSE;
      }
    }
  }
  //printf("is same\n");
  return _XMP_N_INT_TRUE;
}

static _XMP_gmv_desc_t* alloc_gmv_desc(_XMP_gmv_desc_t *desc)
{
  _XMP_gmv_desc_t* new_desc = (_XMP_gmv_desc_t*)malloc(sizeof(_XMP_gmv_desc_t));
  new_desc->is_global = desc->is_global;
  new_desc->ndims = desc->ndims;
  new_desc->a_desc = desc->a_desc;
  new_desc->local_data = desc->local_data;
  new_desc->a_dev = desc->a_dev;

  int ndims = desc->ndims;
  if(desc->local_data != NULL){
    int *lb = new_desc->a_lb = (int*)malloc(sizeof(int)*ndims);
    int *ub = new_desc->a_ub = (int*)malloc(sizeof(int)*ndims);
    for(int i = 0; i < ndims; i++){
      lb[i] = desc->a_lb[i];
      ub[i] = desc->a_ub[i];
    }
  }
  if(desc->a_desc != NULL){
    int *lb = new_desc->lb = (int*)malloc(sizeof(int)*ndims);
    int *ub = new_desc->ub = (int*)malloc(sizeof(int)*ndims);
    int *st = new_desc->st = (int*)malloc(sizeof(int)*ndims);
    for(int i = 0; i < ndims; i++){
      lb[i] = desc->lb[i];
      ub[i] = desc->ub[i];
      st[i] = desc->st[i];
    }
  }
  return new_desc;
}

static void free_gmv_desc(_XMP_gmv_desc_t *desc)
{
  if(desc->local_data != NULL){
    free(desc->a_lb);
    free(desc->a_ub);
  }
  if(desc->a_desc != NULL){
    free(desc->lb);
    free(desc->ub);
    free(desc->st);
  }
  free(desc);
}

static void free_gmv_comm(_XACC_gmv_comm_t* comm)
{
  free_gmv_desc(comm->desc_left);
  free_gmv_desc(comm->desc_right);

  /*
  for(int i = comm->num_recvs - 1; i >= 0; i--){
    MPI_Request_free(comm->req_recvs + i);
  }
  for(int i = comm->num_sends - 1; i >= 0; i--){
    MPI_Request_free(comm->req_sends + i);
  }
  free(comm->req_recvs);
  free(comm->req_sends);
  */
  free(comm);
}


static void cache_comm(_XACC_gmv_comm_t *comm)
{
  //printf("add cache\n");
  if(g_comm_cache[g_comm_cache_tail] != NULL){
    free_gmv_comm(g_comm_cache[g_comm_cache_tail]);
  }
  g_comm_cache[g_comm_cache_tail] = comm;
  if(++g_comm_cache_tail == _XACC_NUM_COMM_CACHES){
    g_comm_cache_tail = 0;
  }
}

static _XACC_gmv_comm_t* find_gmv_comm(_XMP_gmv_desc_t *gmv_desc_leftp, _XMP_gmv_desc_t *gmv_desc_rightp)
{
  for(int i = 0; i < _XACC_NUM_COMM_CACHES; i++){
    _XACC_gmv_comm_t* comm = g_comm_cache[i];
    if(comm == NULL) continue;

    if(is_same_desc(comm->desc_left, gmv_desc_leftp) &&
       is_same_desc(comm->desc_right, gmv_desc_rightp)){
      return comm;
    }
  }
  return NULL;
}

static void exec_gmv_comm(_XACC_gmv_comm_t * comm)
{
  if(comm == NULL) return;

  //printf("exec cached comm\n");
  MPI_Request req[_XACC_MAX_NUM_SENDRECVS];

  //printf("comm_num_sends=%d, recvs=%d\n", comm->num_sends, comm->num_recvs);

  for(int i = 0; i < comm->num_recvs; i++){
    MPI_Irecv(comm->recvs[i].buf, comm->recvs[i].count, comm->recvs[i].datatype, comm->recvs[i].rank, _XMP_N_MPI_TAG_GMOVE, comm->recvs[i].comm, req + i);
  }
  for(int i = 0; i < comm->num_sends; i++){
    MPI_Send(comm->sends[i].buf, comm->sends[i].count, comm->sends[i].datatype, comm->sends[i].rank, _XMP_N_MPI_TAG_GMOVE, comm->sends[i].comm);
  }
  MPI_Waitall(comm->num_recvs, req, MPI_STATUSES_IGNORE);
}

static void set_sendrecv_comm(_XACC_sendrecv_comm_t *sendrecv_comm, void *buf, int count, MPI_Datatype datatype, int rank, MPI_Comm comm)
{
  sendrecv_comm->buf = buf;
  sendrecv_comm->count = count;
  sendrecv_comm->datatype = datatype;
  sendrecv_comm->rank = rank;
  sendrecv_comm->comm = comm;
}
