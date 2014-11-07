#include "xmp.h"
#include "xmp_internal.h"
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef OMNI_TARGET_CPU_KCOMPUTER
#include "fj_lapack.h"
/* #include "fjcoll.h" */
#endif

/* #define DEBUG */

extern void _XMP_align_local_idx(long long int global_idx, int *local_idx,
                                 _XMP_array_t *array, int array_axis, int *rank);

static int xmpf_running=0;

#ifdef DEBUG
static void show_all(_XMP_array_t *p)
{
   MPI_Comm *comm;
   int rank;
   int size;
   int i, j, r;

   comm = _XMP_get_execution_nodes()->comm;
   MPI_Comm_rank(*comm, &rank);
   MPI_Comm_size(*comm, &size);
   for(r=0; r<size; r++){
      if(r==rank){
         printf("--- array info (%d) ---\n", _XMP_get_execution_nodes()->comm_rank);
         printf("dim=%d, type=%d, type_size=%d, alloc=%d(%p)\n",
                p->dim, p->type, (int)(p->type_size), (int)(p->is_allocated), p->array_addr_p);
         printf("align: [");
         for(i=0; i<p->dim; i++){
            printf("%d", p->info[i].align_manner);
            if(i != p->dim-1){
               printf(",");
            }
         }
         printf("]\n");
         printf("global: (");
         for(i=0; i<p->dim; i++){
            printf("%d:%d[%d]", p->info[i].ser_lower, p->info[i].ser_upper, p->info[i].ser_size);
            if(i != p->dim-1){
               printf(",");
            }
         }
         printf(")\n");
         printf("dist: (");
         for(i=0; i<p->dim; i++){
            printf("%d:%d:%d[%d]", p->info[i].par_lower, p->info[i].par_upper, p->info[i].par_stride, p->info[i].par_size);
            if(i != p->dim-1){
               printf(",");
            }
         }
         printf(")\n");
         printf("local: (");
         for(i=0; i<p->dim; i++){
            printf("%d:%d:%d[%d]", p->info[i].local_lower, p->info[i].local_upper, p->info[i].local_stride, p->info[i].alloc_size);
            if(i != p->dim-1){
               printf(",");
            }
         }
         printf(")\n");
         printf("shadow: (");
         for(i=0; i<p->dim; i++){
            printf("%d:%d", p->info[i].shadow_size_lo, p->info[i].shadow_size_hi);
            if(i != p->dim-1){
               printf(",");
            }
         }
         printf(")\n");
         printf("offset: [");
         for(i=0; i<p->dim; i++){
            printf("%lld", p->info[i].align_subscript);
            if(i != p->dim-1){
               printf(",");
            }
         }
         printf("]\n");

         printf("regular: [");
         for(i=0; i<p->dim; i++){
            printf("%d", p->info[i].is_regular_chunk);
            if(i != p->dim-1){
               printf(",");
            }
         }
         printf("]\n");

         printf("template chunk: ");
         for(i=0; i<p->align_template->dim; i++){
            printf("[");
            if(p->align_template->chunk[i].dist_manner == _XMP_N_DIST_BLOCK){
               printf("%lld", p->align_template->chunk[i].par_chunk_width);
            } else if(p->align_template->chunk[i].dist_manner == _XMP_N_DIST_CYCLIC ||
                      p->align_template->chunk[i].dist_manner == _XMP_N_DIST_BLOCK_CYCLIC){
               printf("%lld", p->align_template->chunk[i].par_width);
            } else if(p->align_template->chunk[i].dist_manner == _XMP_N_DIST_GBLOCK){
               for(j=0; j<p->align_template->chunk[i].onto_nodes_info->size+1; j++){
                  printf("%lld", p->align_template->chunk[i].mapping_array[j]);
                  if(j != p->align_template->chunk[i].onto_nodes_info->size){
                     printf(",");
                  }
               }
            }
            printf("]");
         }
         printf("\n");

         printf("align template dim: ");
         for(i=0; i<p->dim; i++){
            printf("%d ", p->info[i].align_template_index);
         }
         printf("\n");
         
         fflush(stdout);
      }
      fflush(stdout);
      MPI_Barrier(*comm);
   }
}

static void show_array(_XMP_array_t *src_d, int *pp)
{
   MPI_Comm *comm;
   int rank;
   int size;
   int i, j, k;
   int r;
   int  *a;

   comm = _XMP_get_execution_nodes()->comm;
   MPI_Comm_rank(*comm, &rank);
   MPI_Comm_size(*comm, &size);
   for(r=0; r<size; r++){
      if(r==rank && src_d->is_allocated){
         printf("\n");
         printf("--- rank %d ---\n", _XMP_get_execution_nodes()->comm_rank);
         if(pp){
            a = pp;
         } else {
            a = (int*)(src_d->array_addr_p);
         }
         if(xmpf_running){
            for(i=src_d->info[0].local_lower; i<=src_d->info[0].local_upper; i++){
               for(j=src_d->info[1].local_lower; j<=src_d->info[1].local_upper; j++){
                  k = (j*src_d->info[0].alloc_size+i);
                  printf("%d ", a[k]);
               }
               printf("\n");
            }
         } else {
            for(i=src_d->info[0].local_lower; i<=src_d->info[0].local_upper; i++){
               for(j=src_d->info[1].local_lower; j<=src_d->info[1].local_upper; j++){
                  k = (i*src_d->info[1].alloc_size+j);
                  printf("%d ", a[k]);
               }
               printf("\n");
            }
         }
      }
      fflush(stdout);
      MPI_Barrier(*comm);
   }
}

static void show_array_ij(int *a, int imax, int jmax)
{
   MPI_Comm *comm;
   int rank;
   int size;
   int i, j, k;
   int r;

   comm = _XMP_get_execution_nodes()->comm;
   MPI_Comm_rank(*comm, &rank);
   MPI_Comm_size(*comm, &size);
   for(r=0; r<size; r++){
      if(r==rank){
         printf("\n");
         printf("--- rank %d ---\n", _XMP_get_execution_nodes()->comm_rank);
         if(xmpf_running){
            for(i=0; i<imax; i++){
               for(j=0; j<jmax; j++){
                  k = (imax*j+i);
                  printf("%d ", a[k]);
               }
               printf("\n");
            }
         } else {
            for(i=0; i<imax; i++){
               for(j=0; j<jmax; j++){
                  k = (jmax*i+j);
                  printf("%d ", a[k]);
               }
               printf("\n");
            }
         }
      }
      fflush(stdout);
      MPI_Barrier(*comm);
   }
}
#endif

#if 0
static int g2p(_XMP_array_t *ap, int dim, int index)
{
   _XMP_template_t *t;
   int tdim, tindex;
   int pi=0;
   int i;
   
   t = ap->align_template;
   tdim = ap->info[dim].align_template_index;

   if(tdim >= 0){
      switch(t->chunk[tdim].dist_manner){
      case _XMP_N_DIST_BLOCK:
         tindex = index+ap->info[dim].align_subscript - t->info[tdim].ser_lower;
         pi = tindex/t->chunk[tdim].par_chunk_width;
         break;
      case _XMP_N_DIST_CYCLIC:
         tindex = index+ap->info[dim].align_subscript - t->info[tdim].ser_lower;
         pi = tindex%t->chunk[tdim].onto_nodes_info->size;
         break;
      case _XMP_N_DIST_BLOCK_CYCLIC:
         tindex = index+ap->info[dim].align_subscript - t->info[tdim].ser_lower;
         pi = (tindex/t->chunk[tdim].par_width)%t->chunk[tdim].onto_nodes_info->size;
         break;
      case _XMP_N_DIST_GBLOCK:
         tindex = index+ap->info[dim].align_subscript - t->info[tdim].ser_lower;;
         for(i=0; i<t->chunk[tdim].onto_nodes_info->size; i++){
            if(tindex < (t->chunk[tdim].mapping_array[i+1] - t->chunk[tdim].mapping_array[0]) &&
               t->chunk[tdim].mapping_array[i+1] - t->chunk[tdim].mapping_array[i] > 0){
               pi = i;
               break;
            }
         }
         break;
      default:
         break;
      }
   }

   return pi;
}
#endif

static int g2p_array(_XMP_array_t *ap, int x, int y)
{
   _XMP_template_t *t;
   int txi, tyi;
   int px=0, py=0;
   int xx, yy;
   int res;

   t = ap->align_template;
   txi = ap->info[0].align_template_index;
   tyi = ap->info[1].align_template_index;

   /* 1次元目 */
   /* px = g2p(ap, 0, x); */
   _XMP_align_local_idx(x, &xx, ap, 0, &px);

   /* 2次元目 */
   /* py = g2p(ap, 1, y); */
   _XMP_align_local_idx(y, &yy, ap, 1, &py);

   if(t->onto_nodes->dim < 2){
      res = px+py;
   } else {
      int size=1;
      int i;
      res = 0;
      for(i=0; i<t->onto_nodes->dim; i++){
         if(txi >= 0 && t->chunk[txi].onto_nodes_index == i){
            res += size*px;
         } else if(tyi >= 0 && t->chunk[tyi].onto_nodes_index == i){
            res += size*py;
         /* } else { */
         /*    res += size*t->onto_nodes->info[i].rank; */
         }
         size *= t->onto_nodes->info[i].size;
      }
   }
   return res;
}

#if 0
static int g2l(_XMP_array_t *ap, int dim, int index)
{
   _XMP_template_t *t;
   _XMP_template_chunk_t *chunk;
   int p;
   int tindex;
   int lindex=index;
   int tdim;
   int alsubs_p;
   int alsubs_i;

   t = ap->align_template;
   tdim = ap->info[dim].align_template_index;
   chunk = &(t->chunk[ap->info[dim].align_template_index]);
   p = g2p(ap, dim, index);

   if(tdim >= 0){
      switch(chunk->dist_manner){
      case _XMP_N_DIST_BLOCK:
         tindex = index+ap->info[dim].align_subscript - t->info[tdim].ser_lower;
         lindex = tindex - p*chunk->par_chunk_width;
         if(ap->info[dim].align_subscript){
            alsubs_p = g2p(ap, dim, ap->info[dim].ser_lower);
            if(alsubs_p == p){
               tindex = ap->info[dim].align_subscript;
               alsubs_i = tindex - alsubs_p*chunk->par_chunk_width;
               lindex -= alsubs_i;
            }
         }
         break;
      case _XMP_N_DIST_CYCLIC:
         tindex = index+ap->info[dim].align_subscript - t->info[tdim].ser_lower;
         lindex = tindex/chunk->onto_nodes_info->size;
         if(ap->info[dim].align_subscript){
            alsubs_i = ap->info[dim].align_subscript/chunk->onto_nodes_info->size;
            alsubs_p = ap->info[dim].align_subscript%chunk->onto_nodes_info->size;
            if(alsubs_p > p){
               alsubs_i++;
            }
            lindex -= alsubs_i;
         }
         break;
      case _XMP_N_DIST_BLOCK_CYCLIC:
         tindex = index+ap->info[dim].align_subscript - t->info[tdim].ser_lower;
         lindex = tindex % chunk->par_width
            + (tindex/(chunk->par_width*chunk->onto_nodes_info->size))*chunk->par_width;
         if(ap->info[dim].align_subscript){
            int bn = ap->info[dim].align_subscript/(chunk->par_width*chunk->onto_nodes_info->size);
            int br = ap->info[dim].align_subscript%(chunk->par_width*chunk->onto_nodes_info->size);
            alsubs_p = br/chunk->par_width;
            alsubs_i = bn*chunk->par_width;
            if(alsubs_p > p){
               alsubs_i += chunk->par_width;
            } else if(alsubs_p == p){
               alsubs_i += br%chunk->par_width;
            }
            lindex -= alsubs_i;
         }
         break;
      case _XMP_N_DIST_GBLOCK:
         tindex = index+ap->info[dim].align_subscript - t->info[tdim].ser_lower;
         lindex = tindex - chunk->mapping_array[p]+chunk->mapping_array[0];
         if(ap->info[dim].align_subscript){
            alsubs_p = g2p(ap, dim, ap->info[dim].ser_lower);
            if(alsubs_p == p){
               tindex = ap->info[dim].align_subscript;
               alsubs_i = tindex - chunk->mapping_array[alsubs_p]+chunk->mapping_array[0];
               lindex -= alsubs_i;
            }
         }
         break;
      default:
         lindex = index-ap->info[dim].ser_lower;
         break;
      }
   } else {
      lindex = index - ap->info[dim].ser_lower;
   }
   lindex += ap->info[dim].shadow_size_lo;
   
   return lindex;
}
#endif

static void array_duplicate(_XMP_array_t *dst_d, MPI_Request *send_req, MPI_Request *recv_req)
{
   _XMP_nodes_t *nodes=dst_d->align_template->onto_nodes;;
   int dim0rank=-1, dim1rank=-1, dist_dim=0;
   int dst_alloc_size[2];
   int i, j, k;

   /* allocate check */
   if(dst_d->is_allocated){
      dst_alloc_size[0] = dst_d->info[0].alloc_size;
      dst_alloc_size[1] = dst_d->info[1].alloc_size;
   } else {
      dst_alloc_size[0] = 0;
      dst_alloc_size[1] = 0;
   }

   if(dst_d->info[0].align_manner == _XMP_N_ALIGN_BLOCK ||
      dst_d->info[0].align_manner == _XMP_N_ALIGN_CYCLIC ||
      dst_d->info[0].align_manner == _XMP_N_ALIGN_BLOCK_CYCLIC ||
      dst_d->info[0].align_manner == _XMP_N_ALIGN_GBLOCK){
      dim0rank = dst_d->align_template->chunk[dst_d->info[0].align_template_index].onto_nodes_index;
      dist_dim++;
   }
   if(dst_d->info[1].align_manner == _XMP_N_ALIGN_BLOCK ||
      dst_d->info[1].align_manner == _XMP_N_ALIGN_CYCLIC ||
      dst_d->info[1].align_manner == _XMP_N_ALIGN_BLOCK_CYCLIC ||
      dst_d->info[1].align_manner == _XMP_N_ALIGN_GBLOCK){
      dim1rank = dst_d->align_template->chunk[dst_d->info[1].align_template_index].onto_nodes_index;
      dist_dim++;
   }
   if(dist_dim < nodes->dim){
      int send_rank=0;
      int size=1;
      for(i=0; i<nodes->dim; i++){
         if(dim0rank == i || dim1rank == i){
            send_rank += size*nodes->info[i].rank;
         }
         size *= nodes->info[i].size;
      }
      /* printf("(%d) send_rank %d\n", nodes->comm_rank, send_rank); */
      for(i=0; i<nodes->comm_size; i++){
         send_req[i] = MPI_REQUEST_NULL;
         recv_req[i] = MPI_REQUEST_NULL;
      }
      if(send_rank == nodes->comm_rank){ /* send */
         int recv_rank;
         for(i=0; i<nodes->comm_size; i++){
            if(i == send_rank) continue;
            recv_rank = i;
            size = 1;
            for(j=0; j<nodes->dim; j++){
               k = (recv_rank/size)%nodes->info[j].size;
               if(j == dim0rank || j == dim1rank){
                  if(nodes->info[j].rank != k) {
                     recv_rank = -1;
                     break;
                  }
               }
               size *= nodes->info[j].size;
            }
            if(recv_rank > -1){
               MPI_Isend(dst_d->array_addr_p, dst_d->type_size*dst_alloc_size[0]*dst_alloc_size[1],
                         MPI_BYTE, recv_rank, 99, *(MPI_Comm*)(nodes->comm), &send_req[recv_rank]);
               /* printf(" duplicate send %d -> %d\n", send_rank, recv_rank); */
            }
         }
      } else {               /* recv */
         MPI_Recv(dst_d->array_addr_p, dst_d->type_size*dst_alloc_size[0]*dst_alloc_size[1],
                  MPI_BYTE, send_rank, 99, *(MPI_Comm*)(nodes->comm), MPI_STATUSES_IGNORE);
         /* printf(" duplicate recv %d -> %d\n", send_rank, nodes->comm_rank); */
      }
   }
}


static void xmp_transpose_no_opt(_XMP_array_t *dst_d, _XMP_array_t *src_d, int opt)
{
   MPI_Comm  *exec_comm, *dst_comm, *src_comm;
   MPI_Group exec_grp, dst_grp, src_grp;
   MPI_Request *send_req, *recv_req;
   int  *e2s, *e2d, *e2e, *s2e, *d2e;
   char *addr_p;                /* for array */
   char *send_buf;              /* MPI send buffer */
   char *recv_buf;              /* MPI recv buffer */
   int   send_buf_offset;
   int  *send_size;
   int  *recv_size;
   int  *recv_pos;
   int   dst_alloc_size[2];
   int   src_alloc_size[2];
   int i, j, k, l, m;

#ifdef DEBUG
   show_all(src_d);             /* debug write */
   show_all(dst_d);             /* debug write */
#endif
   
#ifdef DEBUG
   /* show_array(src_d, addr_p);  /\* debug write *\/ */
#endif

   /* allocate check */
   if(dst_d->is_allocated){
      dst_alloc_size[0] = dst_d->info[0].alloc_size;
      dst_alloc_size[1] = dst_d->info[1].alloc_size;
   } else {
      dst_alloc_size[0] = 0;
      dst_alloc_size[1] = 0;
   }
   if(src_d->is_allocated){
      src_alloc_size[0] = src_d->info[0].alloc_size;
      src_alloc_size[1] = src_d->info[1].alloc_size;
   } else {
      src_alloc_size[0] = 0;
      src_alloc_size[1] = 0;
   }

   /* translate_ranks */
   /* printf("execute comm size: %d\n",  _XMP_get_execution_nodes()->comm_size); */
   /* printf("    dst comm size: %d\n",  dst_d->align_template->onto_nodes->comm_size); */
   /* printf("    src comm size: %d\n",  src_d->align_template->onto_nodes->comm_size); */
   e2e = (int*)_XMP_alloc(_XMP_get_execution_nodes()->comm_size*sizeof(int));
   e2s = (int*)_XMP_alloc(_XMP_get_execution_nodes()->comm_size*sizeof(int));
   e2d = (int*)_XMP_alloc(_XMP_get_execution_nodes()->comm_size*sizeof(int));
   s2e = (int*)_XMP_alloc(_XMP_get_execution_nodes()->comm_size*sizeof(int));
   d2e = (int*)_XMP_alloc(_XMP_get_execution_nodes()->comm_size*sizeof(int));

   if(_XMP_get_execution_nodes()->comm_size != dst_d->align_template->onto_nodes->comm_size ||
      _XMP_get_execution_nodes()->comm_size != src_d->align_template->onto_nodes->comm_size){
      exec_comm = (MPI_Comm*)(_XMP_get_execution_nodes()->comm);
      MPI_Comm_group(*exec_comm, &exec_grp);
      for(i=0; i<_XMP_get_execution_nodes()->comm_size; i++){
         e2e[i] = i;
         s2e[i] = MPI_PROC_NULL;
         d2e[i] = MPI_PROC_NULL;
      }
      if(src_d->is_allocated){
         src_comm  = (MPI_Comm*)(src_d->align_template->onto_nodes->comm);
         MPI_Comm_group(*src_comm, &src_grp);
         MPI_Group_translate_ranks(exec_grp, _XMP_get_execution_nodes()->comm_size, e2e,
                                   src_grp, e2s);
      } else {
         for(i=0; i<_XMP_get_execution_nodes()->comm_size; i++){
            e2s[i] = MPI_PROC_NULL;
         }
      }
      if(dst_d->is_allocated){
         dst_comm  = (MPI_Comm*)(dst_d->align_template->onto_nodes->comm);
         MPI_Comm_group(*dst_comm, &dst_grp);
         MPI_Group_translate_ranks(exec_grp, _XMP_get_execution_nodes()->comm_size, e2e,
                                   dst_grp, e2d);
      } else {
         for(i=0; i<_XMP_get_execution_nodes()->comm_size; i++){
            e2d[i] = MPI_PROC_NULL;
         }
      }
      MPI_Allreduce(MPI_IN_PLACE, e2s, _XMP_get_execution_nodes()->comm_size, MPI_INT, MPI_MAX,
                    *exec_comm);
      MPI_Allreduce(MPI_IN_PLACE, e2d, _XMP_get_execution_nodes()->comm_size, MPI_INT, MPI_MAX,
                    *exec_comm);
      for(i=0; i<_XMP_get_execution_nodes()->comm_size; i++){
         if(e2s[i] >= 0){
            s2e[e2s[i]] = i;
         }
         if(e2d[i] >= 0){
            d2e[e2d[i]] = i;
         }
      }
   } else {
      for(i=0; i<_XMP_get_execution_nodes()->comm_size; i++){
         e2e[i] = i;
         e2s[i] = i;
         e2d[i] = i;
         s2e[i] = i;
         d2e[i] = i;
      }
   }
            
#ifdef DEBUG
   if(_XMP_get_execution_nodes()->comm_rank == 0){
      printf("e2s: ");
      for(i=0; i<_XMP_get_execution_nodes()->comm_size; i++){
         printf("%d ", e2s[i]);
      }
      printf("\n");
      printf("s2e: ");
      for(i=0; i<_XMP_get_execution_nodes()->comm_size; i++){
         printf("%d ", s2e[i]);
      }
      printf("\n");
      printf("e2d: ");
      for(i=0; i<_XMP_get_execution_nodes()->comm_size; i++){
         printf("%d ", e2d[i]);
      }
      printf("\n");
      printf("d2e: ");
      for(i=0; i<_XMP_get_execution_nodes()->comm_size; i++){
         printf("%d ", d2e[i]);
      }
      printf("\n");

      if(xmpf_running){
         printf("array mapping src\n");
         for(i=src_d->info[0].ser_lower; i<=src_d->info[0].ser_upper; i++){
            for(j=src_d->info[1].ser_lower; j<=src_d->info[1].ser_upper; j++){
               printf("%d ", s2e[g2p_array(src_d, i, j)]);
            }
            printf("\n");
         }
         printf("array mapping dst\n");
         for(i=dst_d->info[0].ser_lower; i<=dst_d->info[0].ser_upper; i++){
            for(j=dst_d->info[1].ser_lower; j<=dst_d->info[1].ser_upper; j++){
               printf("%d ", d2e[g2p_array(dst_d, i, j)]);
            }
            printf("\n");
         }
      } else {
         printf("array mapping src\n");
         for(i=src_d->info[0].ser_lower; i<=src_d->info[0].ser_upper; i++){
            for(j=src_d->info[1].ser_lower; j<=src_d->info[1].ser_upper; j++){
               printf("%d ", s2e[g2p_array(src_d, i, j)]);
            }
            printf("\n");
         }
         printf("array mapping dst\n");
         for(i=dst_d->info[0].ser_lower; i<=dst_d->info[0].ser_upper; i++){
            for(j=dst_d->info[1].ser_lower; j<=dst_d->info[1].ser_upper; j++){
               printf("%d ", d2e[g2p_array(dst_d, i, j)]);
            }
            printf("\n");
         }
      }
         
      /* for(j=0; j<src_d->info[1].ser_size; j++){ */
      /*    for(i=0; i<src_d->info[0].ser_size; i++){ */
      /*       printf("(%d,%d)[%d] -> (%d,%d)[%d]\n", */
      /*              g_to_l(src_d, 0, i+src_d->info[0].ser_lower), g_to_l(src_d, 1, j+src_d->info[1].ser_lower), */
      /*              s2e[g2p_array(src_d, i+src_d->info[0].ser_lower, j+src_d->info[1].ser_lower)], */
      /*              g_to_l(dst_d, 0, j+dst_d->info[0].ser_lower), g_to_l(dst_d, 1, i+dst_d->info[1].ser_lower), */
      /*              d2e[g2p_array(dst_d, j+dst_d->info[1].ser_lower, i+dst_d->info[0].ser_lower)]); */
      /*    } */
      /* } */
   }
#endif

  /* transpose & pack src array */
   if(src_d->is_allocated){
      send_buf = (char*)_XMP_alloc(src_alloc_size[0]*src_alloc_size[1]*src_d->type_size
                                   *_XMP_get_execution_nodes()->comm_size);
   } else {
      send_buf = NULL;
   }
   
   if(opt &&
      src_d->is_allocated &&
      dst_alloc_size[0]*dst_alloc_size[1]*dst_d->type_size <=
      src_alloc_size[0]*src_alloc_size[1]*src_d->type_size){
      recv_buf = (char*)src_d->array_addr_p;
   } else {
      if(dst_d->is_allocated){
         recv_buf = (char*)_XMP_alloc(dst_alloc_size[0]*dst_alloc_size[1]*dst_d->type_size);
      } else {
         recv_buf = NULL;
      }
   }
   send_size = (int*)_XMP_alloc(_XMP_get_execution_nodes()->comm_size*sizeof(int));
   recv_size = (int*)_XMP_alloc(_XMP_get_execution_nodes()->comm_size*sizeof(int));
   recv_pos = (int*)_XMP_alloc(_XMP_get_execution_nodes()->comm_size*sizeof(int));
   send_req = (MPI_Request*)_XMP_alloc(_XMP_get_execution_nodes()->comm_size*sizeof(MPI_Request));
   recv_req = (MPI_Request*)_XMP_alloc(_XMP_get_execution_nodes()->comm_size*sizeof(MPI_Request));
   send_buf_offset = src_alloc_size[0]*src_alloc_size[1]*src_d->type_size;
   for(m=0; m<_XMP_get_execution_nodes()->comm_size; m++){
      send_size[m] = 0;
      recv_size[m] = 0;
      send_req[m] = MPI_REQUEST_NULL;
      recv_req[m] = MPI_REQUEST_NULL;
   }

   addr_p = (char*)(src_d->array_addr_p);
   if(xmpf_running){                 /* Fortran */
      for(i=src_d->info[0].ser_lower; i<=src_d->info[0].ser_upper; i++){
         for(j=src_d->info[1].ser_lower; j<=src_d->info[1].ser_upper; j++){
            if(s2e[g2p_array(src_d, i, j)] == _XMP_get_execution_nodes()->comm_rank){
               int li, lj, r;
               m = d2e[g2p_array(dst_d, j, i)];
               /* li = g_to_l(src_d, 0, i); */
               _XMP_align_local_idx((long long int)i, &li, src_d, 0, &r);
               /* lj = g_to_l(src_d, 1, j); */
               _XMP_align_local_idx((long long int)j, &lj, src_d, 1, &r);
               k = (lj*src_alloc_size[0]+li)*src_d->type_size;
               memcpy(send_buf+send_buf_offset*m+send_size[m], addr_p+k, src_d->type_size);
               send_size[m] += src_d->type_size;
               /* for(l=0; l<src_d->type_size; l++){ */
               /*    *(send_buf+send_buf_offset*m+send_size[m]) = *(addr_p+k+l); */
               /*    send_size[m]+=1; */
               /* } */
            }
            
            if(d2e[g2p_array(dst_d, j, i)] == _XMP_get_execution_nodes()->comm_rank){
               m = s2e[g2p_array(src_d, i, j)];
               recv_size[m] += src_d->type_size;
            }
         }
      }
   } else {                     /* C */
      for(j=src_d->info[1].ser_lower; j<=src_d->info[1].ser_upper; j++){
         for(i=src_d->info[0].ser_lower; i<=src_d->info[0].ser_upper; i++){
            if(s2e[g2p_array(src_d, i, j)] == _XMP_get_execution_nodes()->comm_rank){
               int li, lj, r;
               m = d2e[g2p_array(dst_d, j, i)];
               /* li = g_to_l(src_d, 0, i); */
               _XMP_align_local_idx((long long int)i, &li, src_d, 0, &r);
               /* lj = g_to_l(src_d, 1, j); */
               _XMP_align_local_idx((long long int)j, &lj, src_d, 1, &r);
               k = (li*src_alloc_size[1]+lj)*src_d->type_size;
               memcpy(send_buf+send_buf_offset*m+send_size[m], addr_p+k, src_d->type_size);
               send_size[m] += src_d->type_size;
               /* for(l=0; l<src_d->type_size; l++){ */
               /*    *(send_buf+send_buf_offset*m+send_size[m]) = *(addr_p+k+l); */
               /*    send_size[m]+=1; */
               /* } */
            }
            
            if(d2e[g2p_array(dst_d, j, i)] == _XMP_get_execution_nodes()->comm_rank){
               m = s2e[g2p_array(src_d, i, j)];
               recv_size[m] += src_d->type_size;
            }
         }
      }
   }

   recv_pos[0] = 0;
   for(i=1; i<_XMP_get_execution_nodes()->comm_size; i++){
      recv_pos[i] = recv_pos[i-1]+recv_size[i-1];
   }
   
#ifdef DEBUG
   for(m=0; m<_XMP_get_execution_nodes()->comm_size; m++){
      if(m == _XMP_get_execution_nodes()->comm_rank){
         printf("send (%2d) ", m);
         for(i=0; i<_XMP_get_execution_nodes()->comm_size; i++){
            printf("%3d ", send_size[i]);
         }
         printf("\n");
         fflush(stdout);
      }
      MPI_Barrier(*(MPI_Comm*)(_XMP_get_execution_nodes()->comm));
   }
   for(m=0; m<_XMP_get_execution_nodes()->comm_size; m++){
      if(m == _XMP_get_execution_nodes()->comm_rank){
         if(m==0) printf("\n");
         printf("recv (%2d) ", m);
         for(i=0; i<_XMP_get_execution_nodes()->comm_size; i++){
            printf("%3d ", recv_size[i]);
         }
         printf("\n");
         fflush(stdout);
      }
      MPI_Barrier(*(MPI_Comm*)(_XMP_get_execution_nodes()->comm));
   }
   for(m=0; m<_XMP_get_execution_nodes()->comm_size; m++){
      if(m == _XMP_get_execution_nodes()->comm_rank){
         int *a;
         if(m==0) printf("\n");
         printf("send_buf (%2d)\n", m);
         for(j=0; j<_XMP_get_execution_nodes()->comm_size; j++){
            a = (int*)(send_buf+j*send_buf_offset);
            printf(" to %d: ", j);
            for(i=0; i<send_size[j]/src_d->type_size; i++){
               printf("%2d ", a[i]);
            }
            printf("\n");
         }
         fflush(stdout);
      }
      MPI_Barrier(*(MPI_Comm*)(_XMP_get_execution_nodes()->comm));
   }
   show_array(src_d, NULL);
#endif

   /* communicate */
   for(i=0; i<_XMP_get_execution_nodes()->comm_size; i++){
      l = (_XMP_get_execution_nodes()->comm_rank+i)%_XMP_get_execution_nodes()->comm_size;
      m = (_XMP_get_execution_nodes()->comm_rank-i+_XMP_get_execution_nodes()->comm_size)%_XMP_get_execution_nodes()->comm_size;
      if(send_size[l] > 0){
         MPI_Isend(&(*(send_buf+send_buf_offset*l)), send_size[l], MPI_CHAR, l, 99,
                   *(MPI_Comm*)(_XMP_get_execution_nodes()->comm), &send_req[l]);
      }
      if(recv_size[m] > 0){
         MPI_Irecv(recv_buf+recv_pos[m], recv_size[m], MPI_CHAR, m, 99,
                   *(MPI_Comm*)(_XMP_get_execution_nodes()->comm), &recv_req[m]);
      }
   }

   MPI_Waitall(_XMP_get_execution_nodes()->comm_size, send_req, MPI_STATUSES_IGNORE);
   MPI_Waitall(_XMP_get_execution_nodes()->comm_size, recv_req, MPI_STATUSES_IGNORE);

   /* unpack dst array */
   addr_p = (char*)(dst_d->array_addr_p);
   if(xmpf_running){                 /* Fortran */
      for(j=dst_d->info[1].ser_lower; j<=dst_d->info[1].ser_upper; j++){
         for(i=dst_d->info[0].ser_lower; i<=dst_d->info[0].ser_upper; i++){
            if(d2e[g2p_array(dst_d, i, j)] == _XMP_get_execution_nodes()->comm_rank){
               int li, lj, r;
               m = s2e[g2p_array(src_d, j, i)];
               /* li = g_to_l(dst_d, 0, i); */
               _XMP_align_local_idx((long long int)i, &li, dst_d, 0, &r);
               /* lj = g_to_l(dst_d, 1, j); */
               _XMP_align_local_idx((long long int)j, &lj, dst_d, 1, &r);
               k = (lj*dst_alloc_size[0]+li)*dst_d->type_size;
               memcpy(addr_p+k, recv_buf+recv_pos[m], dst_d->type_size);
               recv_pos[m] += dst_d->type_size;
               /* for(l=0; l<dst_d->type_size; l++){ */
               /*    *(addr_p+k+l) = *(recv_buf+recv_pos[m]); */
               /*    recv_pos[m]++; */
               /* } */
            }
         }
      }
   } else {                     /* C */
      for(i=dst_d->info[0].ser_lower; i<=dst_d->info[0].ser_upper; i++){
         for(j=dst_d->info[1].ser_lower; j<=dst_d->info[1].ser_upper; j++){
            if(d2e[g2p_array(dst_d, i, j)] == _XMP_get_execution_nodes()->comm_rank){
               int li, lj, r;
               m = s2e[g2p_array(src_d, j, i)];
               /* li = g_to_l(dst_d, 0, i); */
               _XMP_align_local_idx((long long int)i, &li, dst_d, 0, &r);
               /* lj = g_to_l(dst_d, 1, j); */
               _XMP_align_local_idx((long long int)j, &lj, dst_d, 1, &r);
               k = (li*dst_alloc_size[1]+lj)*dst_d->type_size;
               memcpy(addr_p+k, recv_buf+recv_pos[m], dst_d->type_size);
               recv_pos[m] += dst_d->type_size;
               /* for(l=0; l<dst_d->type_size; l++){ */
               /*    *(addr_p+k+l) = *(recv_buf+recv_pos[m]); */
               /*    recv_pos[m]++; */
               /* } */
            }
         }
      }
   }

   /* duplicate */
   array_duplicate(dst_d, send_req, recv_req);

#ifdef DEBUG
   show_array(dst_d, NULL);
#endif

   /* procedure end */
   if(send_buf) _XMP_free(send_buf);
   _XMP_free(send_size);
   _XMP_free(recv_size);
   _XMP_free(send_req);
   _XMP_free(recv_req);
   if(recv_buf && recv_buf != src_d->array_addr_p && recv_buf != dst_d->array_addr_p){
      _XMP_free(recv_buf);
   }
   _XMP_free(recv_pos);
   _XMP_free(e2e);
   _XMP_free(e2s);
   _XMP_free(e2d);
   _XMP_free(s2e);
   _XMP_free(d2e);
}


static void xmp_transpose_alltoall(_XMP_array_t *dst_d, _XMP_array_t *src_d, int opt, int dist_dim)
{
   char *addr_p;
   char *send_buf;
   char *recv_buf;
   int i, j, k, l;
   int offset_size;
   int src_w;
   int dst_w;
   int buf_size;
   int type_size;
   int dst_alloc_size[2];
   int src_alloc_size[2];
   
#ifdef DEBUG
   show_all(src_d);             /* debug write */
   show_all(dst_d);             /* debug write */
#endif
   
   /* allocate check */
   if(dst_d->is_allocated){
      dst_alloc_size[0] = dst_d->info[0].alloc_size;
      dst_alloc_size[1] = dst_d->info[1].alloc_size;
   } else {
      dst_alloc_size[0] = 0;
      dst_alloc_size[1] = 0;
   }
   if(src_d->is_allocated){
      src_alloc_size[0] = src_d->info[0].alloc_size;
      src_alloc_size[1] = src_d->info[1].alloc_size;
   } else {
      src_alloc_size[0] = 0;
      src_alloc_size[1] = 0;
   }

   /* start_collection("other1"); */
   type_size = dst_d->type_size;
   buf_size =
      (dst_d->info[0].local_upper-dst_d->info[0].local_lower+1) * 
      (dst_d->info[1].local_upper-dst_d->info[1].local_lower+1);
   if(opt && buf_size <= dst_alloc_size[0]*dst_alloc_size[1]){
      send_buf = (char*)(dst_d->array_addr_p);
   } else {
      send_buf = (char*)_XMP_alloc(type_size*src_alloc_size[0]*src_alloc_size[1]);
   }
   if(xmpf_running){           /* Fortran */
      if(dist_dim == 0 && dst_d->info[dist_dim].shadow_size_lo == 0 && dst_d->info[dist_dim].shadow_size_hi == 0){
         recv_buf = (char*)(dst_d->array_addr_p);
      } else if(opt && buf_size <= src_alloc_size[0]*src_alloc_size[1]){
         recv_buf = (char*)(src_d->array_addr_p);
      } else {
         recv_buf = (char*)_XMP_alloc(type_size*buf_size);
      }
   } else {
      if(dist_dim == 1 && dst_d->info[dist_dim].shadow_size_lo == 0 && dst_d->info[dist_dim].shadow_size_hi == 0){
         recv_buf = (char*)(dst_d->array_addr_p);
      } else if(opt && buf_size <= src_alloc_size[0]*src_alloc_size[1]){
         recv_buf = (char*)(src_d->array_addr_p);
      } else {
         recv_buf = (char*)_XMP_alloc(type_size*buf_size);
      }
   }

   /* transpose & pack */
   addr_p = (char*)(src_d->array_addr_p);
   offset_size = 
      (dst_d->info[dist_dim].local_upper-dst_d->info[dist_dim].local_lower+1)*
      (src_d->info[dist_dim].local_upper-src_d->info[dist_dim].local_lower+1);
   src_w =
      (src_d->info[dist_dim].local_upper-src_d->info[dist_dim].local_lower+1);
   dst_w = 
      (dst_d->info[dist_dim].local_upper-dst_d->info[dist_dim].local_lower+1);
   /* stop_collection("other1"); */
   if(xmpf_running){           /* Fortran */
      if(dist_dim == 0){
         for(k=0; k<_XMP_get_execution_nodes()->comm_size; k++){
            for(j=0; j<dst_w; j++){
               l = src_d->info[1].local_lower+k*dst_w+j;
               for(i=src_d->info[0].local_lower; i<=src_d->info[0].local_upper; i++){
                  memcpy(send_buf+(k*offset_size+(i-src_d->info[0].local_lower)*dst_w+j)*type_size,
                         addr_p+(l*src_alloc_size[0]+i)*type_size, type_size);
               }
            }
         }
      } else {
         /* for(k=0; k<_XMP_get_execution_nodes()->comm_size; k++){ */
         /*    for(j=src_d->info[1].local_lower; j<=src_d->info[1].local_upper; j++){ */
         /*       for(i=0; i<dst_w; i++){ */
         /*          l = src_d->info[0].local_lower+k*dst_w+i; */
         /*          memcpy(send_buf+(k*offset_size+i*src_w+j-src_d->info[1].local_lower)*type_size, */
         /*                 addr_p+(j*src_alloc_size[0]+l)*type_size, */
         /*                 type_size); */
         /*       } */
         /*    } */
         /* } */
         if(type_size == 16){
            /* start_collection("feast_pack"); */
            double _Complex *buf_p = (double _Complex*)send_buf;
            double _Complex *base_p = (double _Complex*)(addr_p+(src_alloc_size[0]*src_d->info[1].local_lower+src_d->info[0].local_lower)
                                                         *type_size);
            int nblk = 32;
            int dim0_size = src_d->info[0].local_upper-src_d->info[0].local_lower+1;
            int dim1_size = src_d->info[1].local_upper-src_d->info[1].local_lower+1;
            int alloc_size = src_alloc_size[0];
#pragma omp parallel for private(j,i)
            for(j=0; j<dim1_size; j+=nblk){
               for(i=0; i<dim0_size; i+=nblk){
                  for(int ii=i; ii<i+nblk && ii<dim0_size; ii++){
                     for(int jj=j; jj<j+nblk && jj<dim1_size; jj++){
                        buf_p[src_w*ii+jj] = base_p[jj*alloc_size+ii];
                     }
                  }
               }
            }
            /* stop_collection("feast_pack"); */
         } else {
            char *base_p = (char*)(addr_p+(src_alloc_size[0]*src_d->info[1].local_lower+src_d->info[0].local_lower)
                                   *type_size);
            int nblk = 32;
            int dim0_size = src_d->info[0].local_upper-src_d->info[0].local_lower+1;
            int dim1_size = src_d->info[1].local_upper-src_d->info[1].local_lower+1;
            int alloc_size = src_alloc_size[0];
#pragma omp parallel for private(j,i)
            for(j=0; j<dim1_size; j+=nblk){
               for(i=0; i<dim0_size; i+=nblk){
                  for(int ii=i; ii<i+nblk && ii<dim0_size; ii++){
                     for(int jj=j; jj<j+nblk && jj<dim1_size; jj++){
                        memcpy(send_buf+(src_w*ii+jj)*type_size, base_p+(jj*alloc_size+ii)*type_size, type_size);
                     }
                  }
               }
            }
         }
      }
   } else {                     /* C */
      if(dist_dim == 0){
         for(k=0; k<_XMP_get_execution_nodes()->comm_size; k++){
            for(j=src_d->info[0].local_lower; j<=src_d->info[0].local_upper; j++){
               for(i=0; i<dst_w; i++){
                  l = src_d->info[1].local_lower+k*dst_w+i;
                  memcpy(send_buf+(k*offset_size+i*src_w+j-src_d->info[0].local_lower)*type_size,
                         addr_p+(j*src_alloc_size[1]+l)*type_size,
                         type_size);
               }
            }
         }
      } else {
         for(k=0; k<_XMP_get_execution_nodes()->comm_size; k++){
            for(j=0; j<dst_w; j++){
               l = src_d->info[0].local_lower+k*dst_w+j;
               for(i=src_d->info[1].local_lower; i<=src_d->info[1].local_upper; i++){
                  memcpy(send_buf+(k*offset_size+(i-src_d->info[1].local_lower)*dst_w+j)*type_size,
                         addr_p+(l*src_alloc_size[1]+i)*type_size, type_size);
               }
            }
         }
      }
   }

#ifdef DEBUG
   for(k=0; k<_XMP_get_execution_nodes()->comm_size; k++){
      if(k == _XMP_get_execution_nodes()->comm_rank){
         int *a;
         if(k==0) printf("\n");
         printf("send_buf (%2d)\n", k);
         for(j=0; j<_XMP_get_execution_nodes()->comm_size; j++){
            a = (int*)(send_buf+j*offset_size*type_size);
            printf(" to %d: ", j);
            for(i=0; i<src_w*dst_w; i++){
               printf("%2d ", a[i]);
            }
            printf("\n");
         }
         fflush(stdout);
      }
      MPI_Barrier(*(MPI_Comm*)(_XMP_get_execution_nodes()->comm));
   }
#endif

   /* communication */
   /* start_collection("feast_alltoall"); */
   MPI_Alltoall(send_buf, src_w*dst_w*type_size, MPI_CHAR,
                recv_buf, src_w*dst_w*type_size, MPI_CHAR, *((MPI_Comm*)_XMP_get_execution_nodes()->comm));
   /* stop_collection("feast_alltoall"); */

   /* unpack */
   addr_p = (char*)(dst_d->array_addr_p);
   if(xmpf_running){           /* Fortran */
      if(dist_dim == 1){
         /* start_collection("feast_unpack"); */
#pragma omp parallel for private(j,k)
         for(j=dst_d->info[1].local_lower; j<=dst_d->info[1].local_upper; j++){
            for(k=0; k<_XMP_get_execution_nodes()->comm_size; k++){
               memcpy(addr_p+(dst_alloc_size[0]*j+src_w*k+dst_d->info[0].local_lower)*type_size,
                      recv_buf+(k*src_w*dst_w+(j-dst_d->info[1].local_lower)*src_w)*type_size,
                      src_w*type_size);
            }
         }
         /* stop_collection("feast_unpack"); */
      } else if(dst_d->info[0].shadow_size_lo != 0 || dst_d->info[0].shadow_size_hi != 0){
         for(j=dst_d->info[1].local_lower; j<=dst_d->info[1].local_upper; j++){
            memcpy(addr_p+(dst_alloc_size[0]*j+dst_d->info[0].local_lower)*type_size,
                   recv_buf+((j-dst_d->info[1].local_lower)*dst_w)*type_size,
                   src_w*type_size);
         }
      }
   } else {                     /* C */
      if(dist_dim == 0){
         for(j=dst_d->info[0].local_lower; j<=dst_d->info[0].local_upper; j++){
            for(k=0; k<_XMP_get_execution_nodes()->comm_size; k++){
               memcpy(addr_p+(dst_alloc_size[1]*j+src_w*k+dst_d->info[1].local_lower)*type_size,
                      recv_buf+(k*src_w*dst_w+(j-dst_d->info[0].local_lower)*src_w)*type_size,
                      src_w*type_size);
            }
         }
      } else if(dst_d->info[1].shadow_size_lo != 0 || dst_d->info[1].shadow_size_hi != 0){
         for(j=dst_d->info[0].local_lower; j<=dst_d->info[0].local_upper; j++){
            memcpy(addr_p+(dst_alloc_size[1]*j+dst_d->info[1].local_lower)*type_size,
                   recv_buf+((j-dst_d->info[0].local_lower)*dst_w)*type_size,
                   src_w*type_size);
         }
      }
   }

#ifdef DEBUG
   show_array(src_d, NULL);
   show_array(dst_d, NULL);
#endif
   /* start_collection("other2"); */
   if(send_buf != dst_d->array_addr_p && send_buf != src_d->array_addr_p){
      _XMP_free(send_buf);
   }
   if(recv_buf != dst_d->array_addr_p && recv_buf != src_d->array_addr_p){
      _XMP_free(recv_buf);
   }
   /* stop_collection("other2"); */
}


static void xmp_transpose_original(_XMP_array_t *dst_array, _XMP_array_t *src_array, int opt){

  int nnodes;

  int dst_block_dim, src_block_dim;

  void *sendbuf=NULL, *recvbuf=NULL;
  unsigned long long count, bufsize;

  int dst_chunk_size, type_size;
  int src_chunk_size, src_ser_size;

  nnodes = dst_array->align_template->onto_nodes->comm_size;

  // 2-dimensional Matrix
  if (dst_array->dim != 2) {
    _XMP_fatal("bad dimension for xmp_transpose");
  }

  // No Shadow
  if (dst_array->info[0].shadow_size_lo != 0 ||
      dst_array->info[0].shadow_size_hi != 0 ||
      src_array->info[0].shadow_size_lo != 0 ||
      src_array->info[0].shadow_size_hi != 0) {
   _XMP_fatal("A global array must not have shadows");
  fflush(stdout);
  }

  // Dividable by the number of nodes
  if (dst_array->info[0].ser_size % nnodes != 0) {
   _XMP_fatal("Not dividable by the number of nodes");
  fflush(stdout);
  }

  dst_block_dim = (dst_array->info[0].align_manner == _XMP_N_ALIGN_BLOCK) ? 0 : 1;
  src_block_dim = (src_array->info[0].align_manner == _XMP_N_ALIGN_BLOCK) ? 0 : 1;

  dst_chunk_size = dst_array->info[dst_block_dim].par_size;
  src_chunk_size = src_array->info[src_block_dim].par_size;
  src_ser_size = src_array->info[src_block_dim].ser_size;
  type_size = dst_array->type_size;

  count =  dst_chunk_size * src_chunk_size;
  bufsize = count * nnodes * type_size;

  _XMP_check_reflect_type();

  if (src_block_dim == 1){
    if (opt ==0){
      sendbuf = _XMP_alloc(bufsize);
    }else if (opt==1){
      sendbuf = dst_array->array_addr_p;
    }
    // src_array -> sendbuf
    /* start_collection("org_pack"); */
    _XMP_pack_vector2((char *)sendbuf, (char *)src_array->array_addr_p ,
		      src_chunk_size, dst_chunk_size, nnodes, type_size,
		      src_block_dim);
    /* stop_collection("org_pack"); */
  }
  else {
    sendbuf = src_array->array_addr_p;
  }

  if (opt == 0){
    recvbuf = _XMP_alloc(bufsize);
  }else if (opt ==1){
    recvbuf = src_array->array_addr_p;
  }
  /* start_collection("org_alltoall"); */
  MPI_Alltoall(sendbuf, count * type_size, MPI_BYTE, recvbuf, count * type_size,
               MPI_BYTE, *((MPI_Comm *)src_array->align_template->onto_nodes->comm));
  /* stop_collection("org_alltoall"); */

  if (dst_block_dim == 1){
    /* start_collection("org_unpack"); */
    _XMPF_unpack_transpose_vector((char *)dst_array->array_addr_p ,
       (char *)recvbuf , src_ser_size, dst_chunk_size, type_size, dst_block_dim);
    /* stop_collection("org_unpack"); */

    if (opt==0){
      _XMP_free(recvbuf);
    }
  }

  if (src_block_dim == 1){
    if (opt == 0){
      _XMP_free(sendbuf);
    }
  }


  return;
}


int check_template(_XMP_template_t *dst_t, _XMP_template_t *src_t)
{
   int i, j;
   
   if(dst_t == src_t) return 1;

   if(dst_t->dim != src_t->dim ||
      dst_t->onto_nodes != src_t->onto_nodes ||
      !dst_t->is_fixed || !src_t->is_fixed) return 0;
   
   for(i=0; i<dst_t->dim; i++){
      if(dst_t->chunk[i].par_lower != src_t->chunk[i].par_lower ||
         dst_t->chunk[i].par_upper != src_t->chunk[i].par_upper ||
         dst_t->chunk[i].par_width != src_t->chunk[i].par_width ||
         dst_t->chunk[i].par_stride != src_t->chunk[i].par_stride ||
         dst_t->chunk[i].par_chunk_width != src_t->chunk[i].par_chunk_width ||
         dst_t->chunk[i].dist_manner != src_t->chunk[i].dist_manner ||
         dst_t->chunk[i].onto_nodes_index != src_t->chunk[i].onto_nodes_index) return 0;
      if(dst_t->chunk[i].dist_manner == _XMP_N_DIST_GBLOCK){
         for(j=0; j<=dst_t->onto_nodes->info[dst_t->chunk[i].onto_nodes_index].size; j++){
            if(dst_t->chunk[i].mapping_array[j] != src_t->chunk[i].mapping_array[j]) return 0;
         }
      }
   }
   
   return 1;
}


void xmp_transpose(void *dst_p, void *src_p, int opt)
{
   _XMP_array_t *dst_d;
   _XMP_array_t *src_d;
   int same_nodes;
   int same_template;
   int same_align;
   int dist_num;
   int dist_dim;
   int regular;
   int duplicate;
   int dst_alloc_size[2];
   int src_alloc_size[2];
   int i, j, k;

   dst_d = (_XMP_array_t*)dst_p;
   src_d = (_XMP_array_t*)src_p;

   /* error check */
   if(dst_d->dim != 2 || src_d->dim != 2){
      _XMP_fatal("xmp_transpose: argument dimension is not 2");
      return;
   }
   if(dst_d->type != src_d->type){
      _XMP_fatal("xmp_transpose: argument type is not match");
      return;
   }
   if(!dst_d->align_template->is_distributed || !src_d->align_template->is_distributed){
      _XMP_fatal("xmp_transpose: argument is not distributed");
      return;
   }

   /* allocate check */
   if(dst_d->is_allocated){
      dst_alloc_size[0] = dst_d->info[0].alloc_size;
      dst_alloc_size[1] = dst_d->info[1].alloc_size;
   } else {
      dst_alloc_size[0] = 0;
      dst_alloc_size[1] = 0;
   }
   if(src_d->is_allocated){
      src_alloc_size[0] = src_d->info[0].alloc_size;
      src_alloc_size[1] = src_d->info[1].alloc_size;
   } else {
      src_alloc_size[0] = 0;
      src_alloc_size[1] = 0;
   }

   /* same nodes? */
   same_nodes=1;
   if(_XMP_get_execution_nodes()->comm_size != dst_d->align_template->onto_nodes->comm_size) same_nodes = 0;
   if(_XMP_get_execution_nodes()->comm_size != src_d->align_template->onto_nodes->comm_size) same_nodes = 0;

   /* duplicate? */
   duplicate = 0;
   for(i=0; i<dst_d->dim; i++){
      if(dst_d->info[i].align_template_index >= 0){
         duplicate++;
      }
   }
   if(duplicate >= dst_d->align_template->onto_nodes->dim) duplicate = 0;
      

   /* same template? */
   same_template = check_template(dst_d->align_template, src_d->align_template);

   /* same align? */
   same_align = 1;
   if(dst_d->info[0].align_template_index != src_d->info[0].align_template_index) same_align = 0;
   if(dst_d->info[1].align_template_index != src_d->info[1].align_template_index) same_align = 0;

   /* distribute num & regular */
   dist_num = 0;
   dist_dim = -1;
   regular = 0;
   for(i=0; i<src_d->dim; i++){
      j = src_d->info[i].align_template_index;
      if(j >= 0){
         switch(src_d->align_template->chunk[j].dist_manner){
         case _XMP_N_DIST_BLOCK:
            if(same_align && src_d->info[i].align_manner == dst_d->info[i].align_manner){
               if(dist_num == 0 &&
                  src_d->info[i].ser_size == src_d->align_template->info[j].ser_size &&
                  src_d->info[i].ser_size%_XMP_get_execution_nodes()->comm_size == 0 &&
                  dst_d->info[i].ser_size == dst_d->align_template->info[j].ser_size &&
                  dst_d->info[i].ser_size%_XMP_get_execution_nodes()->comm_size == 0) {
                  regular = 1;
               } else {
                  regular = 0;
               }
            }
            dist_num++;
            dist_dim = i;
            break;
         case _XMP_N_DIST_CYCLIC:
         case _XMP_N_DIST_BLOCK_CYCLIC:
            dist_num++;
            dist_dim = i;
            break;
         case _XMP_N_DIST_GBLOCK:
            if(same_align && src_d->info[i].align_manner == dst_d->info[i].align_manner){
               _XMP_template_chunk_t *src_c = &(src_d->align_template->chunk[src_d->info[i].align_template_index]);
               _XMP_template_chunk_t *dst_c = &(dst_d->align_template->chunk[dst_d->info[i].align_template_index]);
               unsigned long long w;
               if(dist_num == 0 &&
                  src_d->info[i].ser_size == src_d->align_template->info[j].ser_size &&
                  dst_d->info[i].ser_size == dst_d->align_template->info[j].ser_size){
                  regular = 1;

                  w=src_c->mapping_array[1]-src_c->mapping_array[0];
                  for(k=1; k<=src_c->onto_nodes_info->size; k++){
                     if((src_c->mapping_array[k]-src_c->mapping_array[k-1]) != w) regular = 0;
                  }
                  w=dst_c->mapping_array[1]-dst_c->mapping_array[0];
                  for(k=1; k<=dst_c->onto_nodes_info->size; k++){
                     if((dst_c->mapping_array[k]-dst_c->mapping_array[k-1]) != w) regular = 0;
                  }
               }
            }
            dist_num++;
            dist_dim = i;
            break;
         default:
            break;
         }
      }
   }
   /* regular? */
   if(src_d->info[0].align_subscript != 0 || src_d->info[1].align_subscript != 0 ||
      dst_d->info[0].align_subscript != 0 || dst_d->info[1].align_subscript != 0 ) regular = 0;
   
#ifdef DEBUG
   for(i=0; i<_XMP_get_execution_nodes()->comm_size; i++){
      if(i == _XMP_get_execution_nodes()->comm_rank){
         printf("rank%d: nodes %d: template %d: align %d: regular %d\n",
                i, same_nodes, same_template, same_align, regular);
         fflush(stdout);
      }
      fflush(stdout);
      MPI_Barrier(*(MPI_Comm*)(_XMP_get_execution_nodes()->comm));
   }
#endif

   /*============================================================================================*/
   /* same_node && same_template && !same_align && dist_num==1 : no MPI communication            */
   /* same_node && same_align && regular && dist_num==1 : use collective MPI communication       */
   /* other: pack/unpack + send/recv                                                             */
   /*============================================================================================*/
   if(same_nodes && same_template && !same_align && dist_num == 1 &&
      dst_d->info[0].align_subscript == src_d->info[1].align_subscript &&
      dst_d->info[1].align_subscript == src_d->info[0].align_subscript){
      /* no MPI communication transpose */
      int di, dj;
      char *dst_array_p = (char*)dst_d->array_addr_p;
      char *src_array_p = (char*)src_d->array_addr_p;
      
#ifdef DEBUG
      show_all(src_d);
      show_all(dst_d);
#endif
      if(xmpf_running){                 /* Fortran */
         if(dst_d->is_allocated && src_d->is_allocated){
            for(j=src_d->info[1].local_lower; j<=src_d->info[1].local_upper; j++){
               di = j-src_d->info[1].local_lower+dst_d->info[0].local_lower;
               for(i=src_d->info[0].local_lower; i<=src_d->info[0].local_upper; i++){
                  dj = i-src_d->info[0].local_lower+dst_d->info[1].local_lower;
                  memcpy(dst_array_p+(dj*dst_alloc_size[0]+di)*dst_d->type_size,
                         src_array_p+(j*src_alloc_size[0]+i)*src_d->type_size,
                         src_d->type_size);
               }
            }
         }
      } else {                           /* C */
         if(dst_d->is_allocated && src_d->is_allocated){
            for(i=src_d->info[0].local_lower; i<=src_d->info[0].local_upper; i++){
               dj = i-src_d->info[0].local_lower+dst_d->info[1].local_lower;
               for(j=src_d->info[1].local_lower; j<=src_d->info[1].local_upper; j++){
                  di = j-src_d->info[1].local_lower+dst_d->info[0].local_lower;
                  memcpy(dst_array_p+(di*dst_alloc_size[1]+dj)*dst_d->type_size,
                         src_array_p+(i*src_alloc_size[1]+j)*src_d->type_size,
                         src_d->type_size);
               }
            }
         }
      }
#ifdef DEBUG
      show_array(src_d, NULL);
      show_array(dst_d, NULL);
#endif

   } else if(xmpf_running && same_nodes && same_align && regular && !duplicate &&
             dist_num == 1 && dist_dim == 1 && dst_d->info[dist_dim].align_manner == _XMP_N_ALIGN_BLOCK &&
             dst_d->info[dist_dim].shadow_size_lo == 0 && dst_d->info[dist_dim].shadow_size_hi == 0 &&
             src_d->info[dist_dim].shadow_size_lo == 0 && src_d->info[dist_dim].shadow_size_hi == 0){
      /* original xmp_transpose (Fortran) */
      /* start_collection("xmp_transpose_original"); */
      xmp_transpose_original(dst_d, src_d, opt);
      /* stop_collection("xmp_transpose_original"); */
      
   } else if(same_nodes && same_align && regular && !duplicate && dist_num == 1){
      /* collective MPI communication transpose */
      /* start_collection("xmp_transpose_alltoall"); */
      xmp_transpose_alltoall(dst_d, src_d, opt, dist_dim);
      /* stop_collection("xmp_transpose_alltoall"); */

   /* } else if(same_nodes && same_align && !duplicate && dist_num == 1){ */
      /* TODO: not support (block size is not even) */
   /*    /\* collective MPI communication transpose *\/ */
   /*    xmp_transpose_alltoallv(dst_d, src_d, opt, dist_dim); */
      
   } else {
      /* pack/unpack + sendrecv */
      xmp_transpose_no_opt(dst_d, src_d, opt);
   }
}


void xmpf_transpose(void *dst_p, void *src_p, int opt)
{
   xmpf_running = 1;
   xmp_transpose(dst_p, src_p, opt);
   xmpf_running = 0;
}


static int d2p(_XMP_array_t *array_d, int dim)
{
   _XMP_nodes_t *nodes = array_d->align_template->onto_nodes;
   int ret=0;
   int ti;
   int t0, t1;
   int adim=0;

   t0 = array_d->info[0].align_template_index;
   t1 = array_d->info[1].align_template_index;
   if(t0 >= 0) adim++;
   if(t1 >= 0) adim++;

   /* duplicate check */
   if(nodes->dim > adim){
      int p0 = -1;
      int p1 = -1;
      if(t0 >= 0){
         p0 = array_d->align_template->chunk[t0].onto_nodes_index;
      }
      if(t1 >= 0){
         p1 = array_d->align_template->chunk[t1].onto_nodes_index;
      }
      for(int i=0; i<nodes->dim; i++){
         if(i == p0 || i==p1) continue;
         if(nodes->info[i].rank != 0){
            return -1;
         }
      }
   }
   
   ti = array_d->info[dim].align_template_index;
   if(ti >= 0){
      if(array_d->align_template->chunk[ti].onto_nodes_index != _XMP_N_NO_ONTO_NODES){
         ret = array_d->align_template->chunk[ti].onto_nodes_info->rank;
      }
   }
   
   return ret;
}

static int proc_size(_XMP_array_t *array_d, int dim)
{
   int ret=1;
   int ti;

   ti = array_d->info[dim].align_template_index;
   if(ti >= 0){
      if(array_d->align_template->chunk[ti].onto_nodes_index != _XMP_N_NO_ONTO_NODES){
         ret = array_d->align_template->chunk[ti].onto_nodes_info->size;
      }
   }
   
   return ret;
}


static int proc_lower(_XMP_array_t *array_d, int dim)
{
   int ret=_XMP_get_execution_nodes()->comm_size;
   int li, proc;

   /* for(int i=array_d->info[dim].ser_lower; i<=array_d->info[dim].ser_upper; i++){ */
   /*    _XMP_align_local_idx((long long int)i, &li, array_d, dim, &proc); */
   /*    if(ret > proc) ret = proc; */
   /* } */

   switch(array_d->info[dim].align_manner){
   case _XMP_N_ALIGN_BLOCK:
   case _XMP_N_ALIGN_GBLOCK:
      _XMP_align_local_idx((long long int)array_d->info[dim].ser_lower, &li, array_d, dim, &ret);
      break;
   case _XMP_N_ALIGN_CYCLIC:
   case _XMP_N_ALIGN_BLOCK_CYCLIC:
      for(int i=array_d->info[dim].ser_lower; i<=array_d->info[dim].ser_upper; i++){
         _XMP_align_local_idx((long long int)i, &li, array_d, dim, &proc);
         if(ret > proc) ret = proc;
      }
      break;
   default:
      ret = 0;
      break;
   }

   return ret;
}


static int proc_upper(_XMP_array_t *array_d, int dim)
{
   int ret=0;
   int li, proc;

   /* for(int i=array_d->info[dim].ser_lower; i<=array_d->info[dim].ser_upper; i++){ */
   /*    _XMP_align_local_idx((long long int)i, &li, array_d, dim, &proc); */
   /*    if(ret < proc) ret = proc; */
   /* } */
   
   switch(array_d->info[dim].align_manner){
   case _XMP_N_ALIGN_BLOCK:
   case _XMP_N_ALIGN_GBLOCK:
      _XMP_align_local_idx((long long int)array_d->info[dim].ser_upper, &li, array_d, dim, &ret);
      break;
   case _XMP_N_ALIGN_CYCLIC:
   case _XMP_N_ALIGN_BLOCK_CYCLIC:
      for(int i=array_d->info[dim].ser_lower; i<=array_d->info[dim].ser_upper; i++){
         _XMP_align_local_idx((long long int)i, &li, array_d, dim, &proc);
         if(ret < proc) ret = proc;
      }
      break;
   default:
      ret = 0;
      break;
   }
   
   return ret;
}


static int proc_rank(_XMP_array_t *array_d, int idx0, int idx1)
{
   _XMP_nodes_t *nodes_d = array_d->align_template->onto_nodes;
   int p_idx0 = -1;
   int p_idx1 = -1;
   int i, size;
   int ret=0;

   if(array_d->info[0].align_template_index >= 0){
      int ti=array_d->info[0].align_template_index;
      if(array_d->align_template->chunk[ti].onto_nodes_index != _XMP_N_NO_ONTO_NODES){
         p_idx0 = array_d->align_template->chunk[ti].onto_nodes_index;
      }
   }
   if(array_d->info[1].align_template_index >= 0){
      int ti=array_d->info[1].align_template_index;
      if(array_d->align_template->chunk[ti].onto_nodes_index != _XMP_N_NO_ONTO_NODES){
         p_idx1 = array_d->align_template->chunk[ti].onto_nodes_index;
      }
   }

   size = 1;
   for(i=0; i<nodes_d->dim; i++){
      if(p_idx0 == i){
         ret += size*idx0;
      } else if(p_idx1 == i){
         ret += size*idx1;
      /* } else { */
      /*    ret += size*nodes_d->info[i].rank; */
      }
      size *= nodes_d->info[i].size;
   }

   return ret;
}


static int l2g(_XMP_array_t *array_d, int dim, int idx)
{
   _XMP_template_chunk_t *chunk;
   int lidx = idx-array_d->info[dim].local_lower;
   int ret;

   /* switch(array_d->info[dim].align_manner){ */
   /* case _XMP_N_ALIGN_BLOCK: */
   /*    chunk = &(array_d->align_template->chunk[array_d->info[dim].align_template_index]); */
   /*    if(chunk->onto_nodes_info->rank == 0) lidx += array_d->info[dim].align_subscript; */
   /*    ret = lidx+chunk->onto_nodes_info->rank*chunk->par_chunk_width+array_d->info[dim].ser_lower; */
   /*    break; */
   /* case _XMP_N_ALIGN_CYCLIC: */
   /*    chunk = &(array_d->align_template->chunk[array_d->info[dim].align_template_index]); */
   /*    if(chunk->onto_nodes_info->rank == 0) lidx += array_d->info[dim].align_subscript; */
   /*    ret = chunk->onto_nodes_info->size*lidx+chunk->onto_nodes_info->rank+array_d->info[dim].ser_lower; */
   /*    break; */
   /* case _XMP_N_ALIGN_BLOCK_CYCLIC: */
   /*    chunk = &(array_d->align_template->chunk[array_d->info[dim].align_template_index]); */
   /*    if(chunk->onto_nodes_info->rank == 0) lidx += array_d->info[dim].align_subscript; */
   /*    ret = chunk->onto_nodes_info->size*chunk->par_width*(lidx/chunk->par_width) */
   /*       + chunk->onto_nodes_info->rank*chunk->par_width+lidx%chunk->par_width+array_d->info[dim].ser_lower; */
   /*    break; */
   /* case _XMP_N_ALIGN_GBLOCK: */
   /*    chunk = &(array_d->align_template->chunk[array_d->info[dim].align_template_index]); */
   /*    if(chunk->onto_nodes_info->rank == 0) lidx += array_d->info[dim].align_subscript; */
   /*    ret = lidx+chunk->mapping_array[chunk->onto_nodes_info->rank]; */
   /*    break; */
   /* default: */
   /*    return idx+array_d->info[dim].ser_lower; */
   /* } */

   /* ret -= array_d->info[dim].align_subscript; */
   
   switch(array_d->info[dim].align_manner){
   case _XMP_N_ALIGN_BLOCK:
      ret = lidx+array_d->info[dim].par_lower;
      break;
   case _XMP_N_ALIGN_CYCLIC:
      chunk = &(array_d->align_template->chunk[array_d->info[dim].align_template_index]);
      ret = lidx*chunk->onto_nodes_info->size+array_d->info[dim].par_lower;
      break;
   case _XMP_N_ALIGN_BLOCK_CYCLIC:
      chunk = &(array_d->align_template->chunk[array_d->info[dim].align_template_index]);
      if(array_d->info[dim].align_subscript){
         int rank;
         int offset = array_d->info[dim].align_subscript%chunk->par_width;
         rank = (array_d->info[dim].align_subscript/chunk->par_width)%chunk->onto_nodes_info->size;
         if(rank == chunk->onto_nodes_info->rank){
            lidx += offset;
            ret = array_d->info[dim].par_lower+lidx%chunk->par_width
               +(lidx/chunk->par_width)*chunk->onto_nodes_info->size*chunk->par_width;
            ret -= offset;
         } else {
            ret = array_d->info[dim].par_lower+lidx%chunk->par_width
               +(lidx/chunk->par_width)*chunk->onto_nodes_info->size*chunk->par_width;
         }
      } else {
         ret = array_d->info[dim].par_lower+lidx%chunk->par_width
            +(lidx/chunk->par_width)*chunk->onto_nodes_info->size*chunk->par_width;
      }
      break;
   case _XMP_N_ALIGN_GBLOCK:
      ret = lidx+array_d->info[dim].par_lower;
      break;
   default:
      return idx+array_d->info[dim].ser_lower;
   }

   return ret;
}


static void var_mul(_XMP_array_t *x_d, char *x_p, char *a_p, char *b_p)
{
   switch(x_d->type){
   case _XMP_N_TYPE_CHAR:
      *x_p = *a_p * *b_p;
      break;
   case _XMP_N_TYPE_UNSIGNED_CHAR:
      {
         unsigned char *x_var = (unsigned char*)x_p;
         unsigned char *a_var = (unsigned char*)a_p;
         unsigned char *b_var = (unsigned char*)b_p;
         *x_var += *a_var * *b_var;
      }
      break;
   case _XMP_N_TYPE_SHORT:
      {
         short *x_var = (short*)x_p;
         short *a_var = (short*)a_p;
         short *b_var = (short*)b_p;
         *x_var += *a_var * *b_var;
      }
      break;
   case _XMP_N_TYPE_UNSIGNED_SHORT:
      {
         unsigned short *x_var = (unsigned short*)x_p;
         unsigned short *a_var = (unsigned short*)a_p;
         unsigned short *b_var = (unsigned short*)b_p;
         *x_var += *a_var * *b_var;
      }
      break;
   case _XMP_N_TYPE_INT:
      {
         int *x_var = (int*)x_p;
         int *a_var = (int*)a_p;
         int *b_var = (int*)b_p;
         *x_var += *a_var * *b_var;
      }
      break;
   case _XMP_N_TYPE_UNSIGNED_INT:
      {
         unsigned int *x_var = (unsigned int*)x_p;
         unsigned int *a_var = (unsigned int*)a_p;
         unsigned int *b_var = (unsigned int*)b_p;
         *x_var += *a_var * *b_var;
      }
      break;
   case _XMP_N_TYPE_LONG:
      {
         long *x_var = (long*)x_p;
         long *a_var = (long*)a_p;
         long *b_var = (long*)b_p;
         *x_var += *a_var * *b_var;
      }
      break;
   case _XMP_N_TYPE_UNSIGNED_LONG:
      {
         unsigned long *x_var = (unsigned long*)x_p;
         unsigned long *a_var = (unsigned long*)a_p;
         unsigned long *b_var = (unsigned long*)b_p;
         *x_var += *a_var * *b_var;
      }
      break;
   case _XMP_N_TYPE_LONGLONG:
      {
         long long *x_var = (long long*)x_p;
         long long *a_var = (long long*)a_p;
         long long *b_var = (long long*)b_p;
         *x_var += *a_var * *b_var;
      }
      break;
   case _XMP_N_TYPE_UNSIGNED_LONGLONG:
      {
         unsigned long long *x_var = (unsigned long long*)x_p;
         unsigned long long *a_var = (unsigned long long*)a_p;
         unsigned long long *b_var = (unsigned long long*)b_p;
         *x_var += *a_var * *b_var;
      }
      break;
   case _XMP_N_TYPE_FLOAT:
      {
         float *x_var = (float*)x_p;
         float *a_var = (float*)a_p;
         float *b_var = (float*)b_p;
         *x_var += *a_var * *b_var;
      }
      break;
   case _XMP_N_TYPE_DOUBLE:
      {
         double *x_var = (double*)x_p;
         double *a_var = (double*)a_p;
         double *b_var = (double*)b_p;
         *x_var += *a_var * *b_var;
      }
      break;
   case _XMP_N_TYPE_LONG_DOUBLE:
      {
         long double *x_var = (long double*)x_p;
         long double *a_var = (long double*)a_p;
         long double *b_var = (long double*)b_p;
         *x_var += *a_var * *b_var;
      }
      break;
   case _XMP_N_TYPE_FLOAT_IMAGINARY:
      {
         float *x_var = (float*)x_p;
         float *a_var = (float*)a_p;
         float *b_var = (float*)b_p;
         *x_var -= *a_var * *b_var;
      }
      break;
   case _XMP_N_TYPE_DOUBLE_IMAGINARY:
      {
         double *x_var = (double*)x_p;
         double *a_var = (double*)a_p;
         double *b_var = (double*)b_p;
         *x_var -= *a_var * *b_var;
      }
      break;
   case _XMP_N_TYPE_LONG_DOUBLE_IMAGINARY:
      {
         long double *x_var = (long double*)x_p;
         long double *a_var = (long double*)a_p;
         long double *b_var = (long double*)b_p;
         *x_var -= *a_var * *b_var;
      }
      break;
   case _XMP_N_TYPE_FLOAT_COMPLEX:
      {
         float _Complex *x_var = (float _Complex*)x_p;
         float _Complex *a_var = (float _Complex*)a_p;
         float _Complex *b_var = (float _Complex*)b_p;
         *x_var += *a_var * *b_var;
      }
      break;
   case _XMP_N_TYPE_DOUBLE_COMPLEX:
      {
         double _Complex *x_var = (double _Complex*)x_p;
         double _Complex *a_var = (double _Complex*)a_p;
         double _Complex *b_var = (double _Complex*)b_p;
         *x_var += *a_var * *b_var;
      }
      break;
   case _XMP_N_TYPE_LONG_DOUBLE_COMPLEX:
      {
         long double _Complex *x_var = (long double _Complex*)x_p;
         long double _Complex *a_var = (long double _Complex*)a_p;
         long double _Complex *b_var = (long double _Complex*)b_p;
         *x_var += *a_var * *b_var;
      }
      break;
   default:
      break;
   }
}


static void xmp_matmul_no_opt(_XMP_array_t *x_d, _XMP_array_t *a_d, _XMP_array_t *b_d)
{
   MPI_Comm *exec_comm;
   MPI_Request *send_req, *recv_req;
   int *e2e, *e2x, *e2a, *e2b, *x2e, *a2e, *b2e;
   int *send_size, *send_pos;
   int *a_recv_size, *a_recv_pos;
   int *b_recv_size, *b_recv_pos;
   int  buf_offset;
   char *send_buf, *a_buf, *b_buf;
   int  x_alloc_size[2];
   int  a_alloc_size[2];
   int  b_alloc_size[2];
   int i, j, k;
   
#ifdef DEBUG
   show_all(x_d);             /* debug write */
#endif
   
   exec_comm = (MPI_Comm*)(_XMP_get_execution_nodes()->comm);

   /* allocate check */
   if(x_d->is_allocated){
      x_alloc_size[0] = x_d->info[0].alloc_size;
      x_alloc_size[1] = x_d->info[1].alloc_size;
   } else {
      x_alloc_size[0] = 0;
      x_alloc_size[1] = 0;
   }
   if(a_d->is_allocated){
      a_alloc_size[0] = a_d->info[0].alloc_size;
      a_alloc_size[1] = a_d->info[1].alloc_size;
   } else {
      a_alloc_size[0] = 0;
      a_alloc_size[1] = 0;
   }
   if(b_d->is_allocated){
      b_alloc_size[0] = b_d->info[0].alloc_size;
      b_alloc_size[1] = b_d->info[1].alloc_size;
   } else {
      b_alloc_size[0] = 0;
      b_alloc_size[1] = 0;
   }

   /* translate ranks */
   e2e = (int*)_XMP_alloc(_XMP_get_execution_nodes()->comm_size*sizeof(int));
   e2x = (int*)_XMP_alloc(_XMP_get_execution_nodes()->comm_size*sizeof(int));
   e2a = (int*)_XMP_alloc(_XMP_get_execution_nodes()->comm_size*sizeof(int));
   e2b = (int*)_XMP_alloc(_XMP_get_execution_nodes()->comm_size*sizeof(int));
   x2e = (int*)_XMP_alloc(_XMP_get_execution_nodes()->comm_size*sizeof(int));
   a2e = (int*)_XMP_alloc(_XMP_get_execution_nodes()->comm_size*sizeof(int));
   b2e = (int*)_XMP_alloc(_XMP_get_execution_nodes()->comm_size*sizeof(int));

   if(_XMP_get_execution_nodes()->comm_size != x_d->align_template->onto_nodes->comm_size ||
      _XMP_get_execution_nodes()->comm_size != a_d->align_template->onto_nodes->comm_size ||
      _XMP_get_execution_nodes()->comm_size != b_d->align_template->onto_nodes->comm_size){
      MPI_Group exec_grp;
      MPI_Comm_group(*exec_comm, &exec_grp);
      for(i=0; i<_XMP_get_execution_nodes()->comm_size; i++){
         e2e[i] = i;
         x2e[i] = MPI_PROC_NULL;
         a2e[i] = MPI_PROC_NULL;
         b2e[i] = MPI_PROC_NULL;
      }
      if(x_d->is_allocated){
         MPI_Comm *x_comm = (MPI_Comm*)(x_d->align_template->onto_nodes->comm);
         MPI_Group x_grp;
         MPI_Comm_group(*x_comm, &x_grp);
         MPI_Group_translate_ranks(exec_grp, _XMP_get_execution_nodes()->comm_size, e2e,
                                   x_grp, e2x);
      } else {
         for(i=0; i<_XMP_get_execution_nodes()->comm_size; i++){
            e2x[i] = MPI_PROC_NULL;
         }
      }
      if(a_d->is_allocated){
         MPI_Comm *a_comm = (MPI_Comm*)(a_d->align_template->onto_nodes->comm);
         MPI_Group a_grp;
         MPI_Comm_group(*a_comm, &a_grp);
         MPI_Group_translate_ranks(exec_grp, _XMP_get_execution_nodes()->comm_size, e2e,
                                   a_grp, e2a);
      } else {
         for(i=0; i<_XMP_get_execution_nodes()->comm_size; i++){
            e2a[i] = MPI_PROC_NULL;
         }
      }
      if(b_d->is_allocated){
         MPI_Comm *b_comm = (MPI_Comm*)(b_d->align_template->onto_nodes->comm);
         MPI_Group b_grp;
         MPI_Comm_group(*b_comm, &b_grp);
         MPI_Group_translate_ranks(exec_grp, _XMP_get_execution_nodes()->comm_size, e2e,
                                   b_grp, e2b);
      } else {
         for(i=0; i<_XMP_get_execution_nodes()->comm_size; i++){
            e2b[i] = MPI_PROC_NULL;
         }
      }
      MPI_Allreduce(MPI_IN_PLACE, e2x, _XMP_get_execution_nodes()->comm_size, MPI_INT, MPI_MAX,
                    *exec_comm);
      MPI_Allreduce(MPI_IN_PLACE, e2a, _XMP_get_execution_nodes()->comm_size, MPI_INT, MPI_MAX,
                    *exec_comm);
      MPI_Allreduce(MPI_IN_PLACE, e2b, _XMP_get_execution_nodes()->comm_size, MPI_INT, MPI_MAX,
                    *exec_comm);
      for(i=0; i<_XMP_get_execution_nodes()->comm_size; i++){
         if(e2x[i] >= 0){
            x2e[e2x[i]] = i;
         }
         if(e2a[i] >= 0){
            a2e[e2a[i]] = i;
         }
         if(e2b[i] >= 0){
            b2e[e2b[i]] = i;
         }
      }
   } else {
      for(i=0; i<_XMP_get_execution_nodes()->comm_size; i++){
         e2e[i] = i;
         e2x[i] = i;
         e2a[i] = i;
         e2b[i] = i;
         x2e[i] = i;
         a2e[i] = i;
         b2e[i] = i;
      }
   }

#ifdef DEBUG
   if(_XMP_get_execution_nodes()->comm_rank == 0){
      printf("e2x: ");
      for(i=0; i<_XMP_get_execution_nodes()->comm_size; i++){
         printf("%d ", e2x[i]);
      }
      printf("\n");
      printf("x2e: ");
      for(i=0; i<_XMP_get_execution_nodes()->comm_size; i++){
         printf("%d ", x2e[i]);
      }
      printf("\n");
      printf("e2a: ");
      for(i=0; i<_XMP_get_execution_nodes()->comm_size; i++){
         printf("%d ", e2a[i]);
      }
      printf("\n");
      printf("a2e: ");
      for(i=0; i<_XMP_get_execution_nodes()->comm_size; i++){
         printf("%d ", a2e[i]);
      }
      printf("\n");
      printf("e2b: ");
      for(i=0; i<_XMP_get_execution_nodes()->comm_size; i++){
         printf("%d ", e2b[i]);
      }
      printf("\n");
      printf("b2e: ");
      for(i=0; i<_XMP_get_execution_nodes()->comm_size; i++){
         printf("%d ", b2e[i]);
      }
      printf("\n");

      printf("array mapping x\n");
      for(i=x_d->info[0].ser_lower; i<=x_d->info[0].ser_upper; i++){
         for(j=x_d->info[1].ser_lower; j<=x_d->info[1].ser_upper; j++){
            printf("%d ", x2e[g2p_array(x_d, i, j)]);
         }
         printf("\n");
      }
      printf("array mapping a\n");
      for(i=a_d->info[0].ser_lower; i<=a_d->info[0].ser_upper; i++){
         for(j=a_d->info[1].ser_lower; j<=a_d->info[1].ser_upper; j++){
            printf("%d ", a2e[g2p_array(a_d, i, j)]);
         }
         printf("\n");
      }
      printf("array mapping b\n");
      for(i=b_d->info[0].ser_lower; i<=b_d->info[0].ser_upper; i++){
         for(j=b_d->info[1].ser_lower; j<=b_d->info[1].ser_upper; j++){
            printf("%d ", b2e[g2p_array(b_d, i, j)]);
         }
         printf("\n");
      }
   }
#endif

   /* allocate buffer */
   if(a_d->is_allocated || b_d->is_allocated){
      buf_offset = (a_alloc_size[0]*a_alloc_size[1] > b_alloc_size[0]*b_alloc_size[1])?
         a_alloc_size[0]*a_alloc_size[1]: b_alloc_size[0]*b_alloc_size[1];
   } else {
      buf_offset = 0;
   }
   send_req = (MPI_Request*)_XMP_alloc(_XMP_get_execution_nodes()->comm_size*sizeof(MPI_Request));
   recv_req = (MPI_Request*)_XMP_alloc(_XMP_get_execution_nodes()->comm_size*sizeof(MPI_Request));
   send_buf = (char*)_XMP_alloc(buf_offset * x_d->type_size * _XMP_get_execution_nodes()->comm_size);
   if(x_d->is_allocated){
      a_buf = (char*)_XMP_alloc(a_d->info[0].ser_size*a_d->info[1].ser_size*a_d->type_size);
      b_buf = (char*)_XMP_alloc(b_d->info[0].ser_size*b_d->info[1].ser_size*b_d->type_size);
   } else {
      a_buf = NULL;
      b_buf = NULL;
   }
   send_pos = (int*)_XMP_alloc(_XMP_get_execution_nodes()->comm_size*sizeof(int));
   send_size = (int*)_XMP_alloc(_XMP_get_execution_nodes()->comm_size*sizeof(int));
   a_recv_pos = (int*)_XMP_alloc(_XMP_get_execution_nodes()->comm_size*sizeof(int));
   a_recv_size = (int*)_XMP_alloc(_XMP_get_execution_nodes()->comm_size*sizeof(int));
   b_recv_pos = (int*)_XMP_alloc(_XMP_get_execution_nodes()->comm_size*sizeof(int));
   b_recv_size = (int*)_XMP_alloc(_XMP_get_execution_nodes()->comm_size*sizeof(int));

   /* array a pack */
#ifdef DEBUG
   show_all(a_d);             /* debug write */
#endif
   for(i=0; i<_XMP_get_execution_nodes()->comm_size; i++){
      send_size[i] = 0;
      a_recv_size[i] = 0;
      send_pos[i] = i * buf_offset * x_d->type_size;
      send_req[i] = MPI_REQUEST_NULL;
      recv_req[i] = MPI_REQUEST_NULL;
   }

   if(xmpf_running){           /* Fortran */
      char *addr_p = (char*)a_d->array_addr_p;
      int cp_size = a_d->type_size;
      int xj, xp;
      int aj, ap;
      for(j=x_d->info[0].ser_lower; j<=x_d->info[0].ser_upper; j++){
         _XMP_align_local_idx((long long int)j, &xj, x_d, 0, &xp);
         _XMP_align_local_idx((long long int)(j-x_d->info[0].ser_lower+a_d->info[0].ser_lower), &aj, a_d, 0, &ap);
         if(a_d->is_allocated && ap == d2p(a_d, 0)){
            for(i=a_d->info[1].local_lower; i<=a_d->info[1].local_upper; i++){
               memcpy(send_buf+send_pos[xp]+send_size[xp],
                      addr_p+(a_alloc_size[0]*i+aj)*a_d->type_size,
                      cp_size);
               send_size[xp] += cp_size;
            }
         }
         if(x_d->is_allocated && xp == d2p(x_d, 0)){
            for(i=a_d->info[1].ser_lower; i<=a_d->info[1].ser_upper; i++){
               a_recv_size[a2e[g2p_array(a_d, j-x_d->info[0].ser_lower+a_d->info[0].ser_lower, i)]] += cp_size;
            }
         }
      }
      
      a_recv_pos[0] = 0;
      for(i=1; i<_XMP_get_execution_nodes()->comm_size; i++){
         a_recv_pos[i] = a_recv_pos[i-1]+a_recv_size[i-1];
      }

      /* array a communicate */
#ifdef DEBUG
      fflush(stdout);
      MPI_Barrier(*(MPI_Comm*)(_XMP_get_execution_nodes()->comm));
#endif
      if(x_d->is_allocated){
         for(i=0; i<proc_size(a_d, 1); i++){
            for(j=0; j<proc_size(a_d, 0); j++){
               int src_rank = proc_rank(a_d, j, i);
               if(a_recv_size[a2e[src_rank]] > 0){
#ifdef DEBUG
                  printf("recv: %d -> %d: %d\n",
                         src_rank, _XMP_get_execution_nodes()->comm_rank, a_recv_size[b2e[src_rank]]);
#endif
                  MPI_Irecv(a_buf+a_recv_pos[a2e[src_rank]], a_recv_size[a2e[src_rank]], MPI_CHAR, a2e[src_rank], 99,
                            *exec_comm, &recv_req[a2e[src_rank]]);
               }
            }
         }
      }
#ifdef DEBUG
      fflush(stdout);
      MPI_Barrier(*(MPI_Comm*)(_XMP_get_execution_nodes()->comm));
#endif
      if(a_d->is_allocated){
         for(i=proc_lower(x_d, 1); i<=proc_upper(x_d, 1); i++){
            for(j=proc_lower(x_d, 0); j<=proc_upper(x_d, 0); j++){
               int dst_rank = proc_rank(x_d, j, i);
               if(send_size[j] > 0){
#ifdef DEBUG
                  printf("send: %d -> %d: %d\n", _XMP_get_execution_nodes()->comm_rank, dst_rank, send_size[j]);
#endif
                  MPI_Isend(send_buf+send_pos[j], send_size[j], MPI_CHAR, x2e[dst_rank], 99,
                            *exec_comm, &send_req[x2e[dst_rank]]);
               }
            }
         }
      }
#ifdef DEBUG
      fflush(stdout);
      MPI_Barrier(*(MPI_Comm*)(_XMP_get_execution_nodes()->comm));
#endif
      MPI_Waitall(_XMP_get_execution_nodes()->comm_size, send_req, MPI_STATUSES_IGNORE);
      MPI_Waitall(_XMP_get_execution_nodes()->comm_size, recv_req, MPI_STATUSES_IGNORE);

   } else {                     /* C */
      char *addr_p = (char*)a_d->array_addr_p;
      int cp_size = (a_d->info[1].local_upper-a_d->info[1].local_lower+1)*a_d->type_size;
      int xj, xp;
      int aj, ap;
      for(j=x_d->info[0].ser_lower; j<=x_d->info[0].ser_upper; j++){
         _XMP_align_local_idx((long long int)j, &xj, x_d, 0, &xp);
         _XMP_align_local_idx((long long int)(j-x_d->info[0].ser_lower+a_d->info[0].ser_lower), &aj, a_d, 0, &ap);
         if(a_d->is_allocated && ap == d2p(a_d, 0)){
            memcpy(send_buf+send_pos[xp]+send_size[xp],
                   addr_p+(a_alloc_size[1]*aj+a_d->info[1].local_lower)*a_d->type_size,
                   cp_size);
            send_size[xp] += cp_size;
         }
         if(x_d->is_allocated && xp == d2p(x_d, 0)){
            for(i=a_d->info[1].ser_lower; i<=a_d->info[1].ser_upper; i++){
               a_recv_size[a2e[g2p_array(a_d, j-x_d->info[0].ser_lower+a_d->info[0].ser_lower, i)]] += a_d->type_size;
            }
         }
      }
      
      a_recv_pos[0] = 0;
      for(i=1; i<_XMP_get_execution_nodes()->comm_size; i++){
         a_recv_pos[i] = a_recv_pos[i-1]+a_recv_size[i-1];
      }

      /* array a communicate */
#ifdef DEBUG
      fflush(stdout);
      MPI_Barrier(*(MPI_Comm*)(_XMP_get_execution_nodes()->comm));
#endif
      if(x_d->is_allocated){
         for(j=0; j<proc_size(a_d, 0); j++){
            for(i=0; i<proc_size(a_d, 1); i++){
               int src_rank = proc_rank(a_d, j, i);
               if(a_recv_size[a2e[src_rank]] > 0){
#ifdef DEBUG
                  printf("recv: %d -> %d: %d\n", src_rank, _XMP_get_execution_nodes()->comm_rank, a_recv_size[a2e[src_rank]]);
#endif
                  MPI_Irecv(a_buf+a_recv_pos[a2e[src_rank]], a_recv_size[a2e[src_rank]], MPI_CHAR, a2e[src_rank], 99,
                            *exec_comm, &recv_req[a2e[src_rank]]);
               }
            }
         }
      }
#ifdef DEBUG
      fflush(stdout);
      MPI_Barrier(*(MPI_Comm*)(_XMP_get_execution_nodes()->comm));
#endif
      if(a_d->is_allocated){
         for(j=proc_lower(x_d, 0); j<=proc_upper(x_d, 0); j++){
            for(i=proc_lower(x_d, 1); i<=proc_upper(x_d, 1); i++){
               int dst_rank = proc_rank(x_d, j, i);
               if(send_size[j] > 0){
#ifdef DEBUG
                  printf("send: %d -> %d: %d\n", _XMP_get_execution_nodes()->comm_rank, dst_rank, send_size[j]);
#endif
                  MPI_Isend(send_buf+send_pos[j], send_size[j], MPI_CHAR, x2e[dst_rank], 99,
                            *exec_comm, &send_req[x2e[dst_rank]]);
               }
            }
         }
      }
#ifdef DEBUG
      fflush(stdout);
      MPI_Barrier(*(MPI_Comm*)(_XMP_get_execution_nodes()->comm));
#endif
      MPI_Waitall(_XMP_get_execution_nodes()->comm_size, send_req, MPI_STATUSES_IGNORE);
      MPI_Waitall(_XMP_get_execution_nodes()->comm_size, recv_req, MPI_STATUSES_IGNORE);
   }
      
#ifdef DEBUG
   show_array(a_d, NULL);
   for(k=0; k<_XMP_get_execution_nodes()->comm_size; k++){
      if(k == _XMP_get_execution_nodes()->comm_rank){
         int *a;
         if(k==0) printf("\n");
         printf("send_buf (%2d)\n", k);
         for(j=0; j<_XMP_get_execution_nodes()->comm_size; j++){
            a = (int*)(send_buf+send_pos[j]);
            printf(" to %d: ", j);
            for(i=0; i<send_size[j]/x_d->type_size; i++){
               printf("%2d ", a[i]);
            }
            printf("(%d)\n", (int)(send_size[j]/x_d->type_size));
         }
         fflush(stdout);
      }
      MPI_Barrier(*(MPI_Comm*)(_XMP_get_execution_nodes()->comm));
   }
   for(k=0; k<_XMP_get_execution_nodes()->comm_size; k++){
      if(k == _XMP_get_execution_nodes()->comm_rank){
         int *a;
         if(k==0) printf("\n");
         printf("recv_buf (%2d)\n", k);
         for(j=0; j<_XMP_get_execution_nodes()->comm_size; j++){
            a = (int*)(a_buf+a_recv_pos[j]);
            printf(" from %d: ", j);
            for(i=0; i<a_recv_size[j]/x_d->type_size; i++){
               printf("%2d ", a[i]);
            }
            printf("(%d)\n", (int)(a_recv_size[j]/x_d->type_size));
         }
         fflush(stdout);
      }
      MPI_Barrier(*(MPI_Comm*)(_XMP_get_execution_nodes()->comm));
   }
#endif

   /* array b pack */
#ifdef DEBUG
   show_all(b_d);             /* debug write */
#endif
   for(i=0; i<_XMP_get_execution_nodes()->comm_size; i++){
      send_size[i] = 0;
      b_recv_size[i] = 0;
      send_pos[i] = i*buf_offset*x_d->type_size;
      send_req[i] = MPI_REQUEST_NULL;
      recv_req[i] = MPI_REQUEST_NULL;
   }

   if(xmpf_running){           /* Fortran */
      char *addr_p = (char*)b_d->array_addr_p;
      int cp_size = (b_d->info[0].local_upper-b_d->info[0].local_lower+1)*b_d->type_size;
      int xj, xp;
      int bj, bp;
      for(j=x_d->info[1].ser_lower; j<=x_d->info[1].ser_upper; j++){
         _XMP_align_local_idx((long long int)j, &xj, x_d, 1, &xp);
         _XMP_align_local_idx((long long int)(j-x_d->info[1].ser_lower+b_d->info[1].ser_lower), &bj, b_d, 1, &bp);
         if(b_d->is_allocated && bp == d2p(b_d, 1)){
            memcpy(send_buf+send_pos[xp]+send_size[xp],
                   addr_p+(b_alloc_size[0]*bj+b_d->info[0].local_lower)*b_d->type_size,
                   cp_size);
            send_size[xp] += cp_size;
         }
         if(x_d->is_allocated && xp == d2p(x_d, 1)){
            for(i=b_d->info[0].ser_lower; i<=b_d->info[0].ser_upper; i++){
               b_recv_size[b2e[g2p_array(b_d, i, j-x_d->info[1].ser_lower+b_d->info[1].ser_lower)]] += b_d->type_size;
            }
         }
      }
      
      b_recv_pos[0] = 0;
      for(i=1; i<_XMP_get_execution_nodes()->comm_size; i++){
         b_recv_pos[i] = b_recv_pos[i-1]+b_recv_size[i-1];
      }

      /* array b communicate */
#ifdef DEBUG
      fflush(stdout);
      MPI_Barrier(*(MPI_Comm*)(_XMP_get_execution_nodes()->comm));
      for(i=0; i<_XMP_get_execution_nodes()->comm_size; i++){
         if(i == _XMP_get_execution_nodes()->comm_rank){
            printf(" rank %d: send ", i);
            for(j=0; j<_XMP_get_execution_nodes()->comm_size; j++){
               printf("%d ", send_size[j]);
            }
            printf(": recv ");
            for(j=0; j<_XMP_get_execution_nodes()->comm_size; j++){
               printf("%d ", b_recv_size[j]);
            }
            printf(": cp_size %d\n", cp_size);
         }
         fflush(stdout);
         MPI_Barrier(*(MPI_Comm*)(_XMP_get_execution_nodes()->comm));
      }
#endif
      if(x_d->is_allocated){
         for(j=0; j<proc_size(b_d, 0); j++){
            for(i=0; i<proc_size(b_d, 1); i++){
               int src_rank = proc_rank(b_d, j, i);
               if(b_recv_size[b2e[src_rank]] > 0){
#ifdef DEBUG
                  printf("recv: %d -> %d: %d\n",
                         src_rank, _XMP_get_execution_nodes()->comm_rank, b_recv_size[b2e[src_rank]]);
#endif
                  MPI_Irecv(b_buf+b_recv_pos[b2e[src_rank]], b_recv_size[b2e[src_rank]], MPI_CHAR, b2e[src_rank], 99,
                            *exec_comm, &recv_req[b2e[src_rank]]);
               }
            }
         }
      }
#ifdef DEBUG
      fflush(stdout);
      MPI_Barrier(*(MPI_Comm*)(_XMP_get_execution_nodes()->comm));
#endif
      if(b_d->is_allocated){
         for(j=proc_lower(x_d, 0); j<=proc_upper(x_d, 0); j++){
            for(i=proc_lower(x_d, 1); i<=proc_upper(x_d, 1); i++){
               int dst_rank = proc_rank(x_d, j, i);
               if(send_size[i] > 0){
#ifdef DEBUG
                  printf("send: %d -> %d: %d\n", _XMP_get_execution_nodes()->comm_rank, dst_rank, send_size[i]);
#endif
                  MPI_Isend(send_buf+send_pos[i], send_size[i], MPI_CHAR, x2e[dst_rank], 99,
                            *exec_comm, &send_req[x2e[dst_rank]]);
               }
            }
         }
      }
#ifdef DEBUG
      fflush(stdout);
      MPI_Barrier(*(MPI_Comm*)(_XMP_get_execution_nodes()->comm));
#endif
      MPI_Waitall(_XMP_get_execution_nodes()->comm_size, send_req, MPI_STATUSES_IGNORE);
      MPI_Waitall(_XMP_get_execution_nodes()->comm_size, recv_req, MPI_STATUSES_IGNORE);

   } else {                     /* C */
      char *addr_p = (char*)b_d->array_addr_p;
      int cp_size = b_d->type_size;
      int xj, xp;
      int bj, bp;
      for(j=x_d->info[1].ser_lower; j<=x_d->info[1].ser_upper; j++){
         _XMP_align_local_idx((long long int)j, &xj, x_d, 1, &xp);
         _XMP_align_local_idx((long long int)(j-x_d->info[1].ser_lower+b_d->info[1].ser_lower), &bj, b_d, 1, &bp);
         if(b_d->is_allocated && bp == d2p(b_d, 1)){
            for(i=b_d->info[0].local_lower; i<=b_d->info[0].local_upper; i++){
               memcpy(send_buf+send_pos[xp]+send_size[xp],
                      addr_p+(b_alloc_size[1]*i+bj)*b_d->type_size,
                      cp_size);
               send_size[xp] += cp_size;
            }
         }
         if(x_d->is_allocated && xp == d2p(x_d, 1)){
            for(i=b_d->info[0].ser_lower; i<=b_d->info[0].ser_upper; i++){
               b_recv_size[b2e[g2p_array(b_d, i, j-x_d->info[1].ser_lower+b_d->info[1].ser_lower)]] += b_d->type_size;
            }
         }
      }
      
      b_recv_pos[0] = 0;
      for(i=1; i<_XMP_get_execution_nodes()->comm_size; i++){
         b_recv_pos[i] = b_recv_pos[i-1]+b_recv_size[i-1];
      }

      /* array b communicate */
#ifdef DEBUG
      fflush(stdout);
      MPI_Barrier(*(MPI_Comm*)(_XMP_get_execution_nodes()->comm));
#endif
      if(x_d->is_allocated){
         for(i=0; i<proc_size(b_d, 1); i++){
            for(j=0; j<proc_size(b_d, 0); j++){
               int src_rank = proc_rank(b_d, j, i);
               if(b_recv_size[b2e[src_rank]] > 0){
#ifdef DEBUG
                  printf("recv: %d -> %d: %d\n",
                         src_rank, _XMP_get_execution_nodes()->comm_rank, b_recv_size[b2e[src_rank]]);
#endif
                  MPI_Irecv(b_buf+b_recv_pos[b2e[src_rank]], b_recv_size[b2e[src_rank]], MPI_CHAR, b2e[src_rank], 99,
                            *exec_comm, &recv_req[b2e[src_rank]]);
               }
            }
         }
      }
#ifdef DEBUG
      fflush(stdout);
      MPI_Barrier(*(MPI_Comm*)(_XMP_get_execution_nodes()->comm));
#endif
      if(b_d->is_allocated){
         for(i=proc_lower(x_d, 1); i<=proc_upper(x_d, 1); i++){
            for(j=proc_lower(x_d, 0); j<=proc_upper(x_d, 0); j++){
               int dst_rank = proc_rank(x_d, j, i);
               if(send_size[i] > 0){
#ifdef DEBUG
                  printf("send: %d -> %d: %d, (%d,%d)\n",
                         _XMP_get_execution_nodes()->comm_rank, dst_rank, send_size[i], j, i);
#endif
                  MPI_Isend(send_buf+send_pos[i], send_size[i], MPI_CHAR, x2e[dst_rank], 99,
                            *exec_comm, &send_req[x2e[dst_rank]]);
               }
            }
         }
      }
#ifdef DEBUG
      fflush(stdout);
      MPI_Barrier(*(MPI_Comm*)(_XMP_get_execution_nodes()->comm));
#endif
      MPI_Waitall(_XMP_get_execution_nodes()->comm_size, send_req, MPI_STATUSES_IGNORE);
      MPI_Waitall(_XMP_get_execution_nodes()->comm_size, recv_req, MPI_STATUSES_IGNORE);
   }

#ifdef DEBUG
   show_array(b_d, NULL);
   for(k=0; k<_XMP_get_execution_nodes()->comm_size; k++){
      if(k == _XMP_get_execution_nodes()->comm_rank){
         int *a;
         if(k==0) printf("\n");
         printf("send_buf (%2d)\n", k);
         for(j=0; j<_XMP_get_execution_nodes()->comm_size; j++){
            a = (int*)(send_buf+send_pos[j]);
            printf(" to %d: ", j);
            for(i=0; i<send_size[j]/x_d->type_size; i++){
               printf("%2d ", a[i]);
            }
            printf("(%d)\n", (int)(send_size[j]/x_d->type_size));
         }
         fflush(stdout);
      }
      fflush(stdout);
      MPI_Barrier(*(MPI_Comm*)(_XMP_get_execution_nodes()->comm));
   }
   for(k=0; k<_XMP_get_execution_nodes()->comm_size; k++){
      if(k == _XMP_get_execution_nodes()->comm_rank){
         int *a;
         if(k==0) printf("\n");
         printf("recv_buf (%2d)\n", k);
         for(j=0; j<_XMP_get_execution_nodes()->comm_size; j++){
            a = (int*)(b_buf+b_recv_pos[j]);
            printf(" from %d: ", j);
            for(i=0; i<b_recv_size[j]/x_d->type_size; i++){
               printf("%2d ", a[i]);
            }
            printf("(%d)\n", (int)(b_recv_size[j]/x_d->type_size));
         }
         fflush(stdout);
      }
      fflush(stdout);
      MPI_Barrier(*(MPI_Comm*)(_XMP_get_execution_nodes()->comm));
   }
#endif

   /* matmul */
   if(x_d->is_allocated){
      if(xmpf_running){           /* Fortran */
         int a_rank0, a_idx0, a_rank1, a_idx1;
         int b_rank0, b_idx0, b_rank1, b_idx1;
         int *a_offset = send_pos;
         int *b_offset = send_size;
      
         for(k=0; k<_XMP_get_execution_nodes()->comm_size; k++){
            b_recv_size[k] = 0;
         }
         for(k=b_d->info[0].ser_lower; k<=b_d->info[0].ser_upper; k++){
            _XMP_align_local_idx((long long int)(k), &b_idx0, b_d, 0, &b_rank0);
            b_recv_size[b_rank0]++;
         }
         for(k=0; k<_XMP_get_execution_nodes()->comm_size; k++){
            b_offset[k] = 0;
         }
         for(i=x_d->info[1].local_lower; i<=x_d->info[1].local_upper; i++){
            int gi = l2g(x_d, 1, i)-x_d->info[1].ser_lower+b_d->info[1].ser_lower;
            _XMP_align_local_idx((long long int)(gi), &b_idx1, b_d, 1, &b_rank1);
            for(k=0; k<_XMP_get_execution_nodes()->comm_size; k++){
               a_offset[k] = 0;
            }
            for(j=x_d->info[0].local_lower; j<=x_d->info[0].local_upper; j++){
               int gj = l2g(x_d, 0, j)-x_d->info[0].ser_lower+a_d->info[0].ser_lower;
               _XMP_align_local_idx((long long int)(gj), &a_idx0, a_d, 0, &a_rank0);
               char *x_p = (char*)(x_d->array_addr_p)+(i*x_alloc_size[0]+j)*x_d->type_size;
               memset(x_p, 0, x_d->type_size);
               for(k=0; k<a_d->info[1].ser_size; k++){
                  _XMP_align_local_idx((long long int)(k+a_d->info[1].ser_lower), &a_idx1, a_d, 1, &a_rank1);
                  _XMP_align_local_idx((long long int)(k+b_d->info[0].ser_lower), &b_idx0, b_d, 0, &b_rank0);
                  char *a_p = (char*)(a_buf+a_recv_pos[a2e[g2p_array(a_d,gj,k+a_d->info[1].ser_lower)]]
                                      +(a_offset[a2e[g2p_array(a_d,gj,k+a_d->info[1].ser_lower)]])*a_d->type_size);
                  char *b_p = (char*)(b_buf+b_recv_pos[b2e[g2p_array(b_d,k+b_d->info[0].ser_lower,gi)]]
                                      +(b_idx0+b_offset[b2e[g2p_array(b_d,k+b_d->info[0].ser_lower,gi)]])*b_d->type_size);
#ifdef DEBUG
                  if(j==x_d->info[0].local_lower+0 && i==x_d->info[1].local_lower+0 &&
                     _XMP_get_execution_nodes()->comm_rank == 0){
                     printf("%4d x %4d: %d %d %d\n", *((int*)a_p), *((int*)b_p),
                            i, l2g(x_d, 1, i), gi);
                  }
#endif
                  a_offset[a2e[g2p_array(a_d,gj,k+a_d->info[1].ser_lower)]]++;
                  var_mul(x_d, x_p, a_p, b_p);
               }
            }
            for(k=0; k<proc_size(b_d, 0); k++){
               b_offset[b2e[proc_rank(b_d, k, b_rank1)]] += b_recv_size[k];
            }
         }
      
      } else {                     /* C */
         int a_rank0, a_idx0, a_rank1, a_idx1;
         int b_rank0, b_idx0, b_rank1, b_idx1;
         int *a_offset = send_pos;
         int *b_offset = send_size;
      
         for(k=0; k<_XMP_get_execution_nodes()->comm_size; k++){
            a_recv_size[k] = 0;
         }
         for(k=a_d->info[1].ser_lower; k<=a_d->info[1].ser_upper; k++){
            _XMP_align_local_idx((long long int)(k), &a_idx1, a_d, 1, &a_rank1);
            a_recv_size[a_rank1]++;
         }
         for(k=0; k<_XMP_get_execution_nodes()->comm_size; k++){
            a_offset[k] = 0;
         }
         for(j=x_d->info[0].local_lower; j<=x_d->info[0].local_upper; j++){
            int gj = l2g(x_d, 0, j)-x_d->info[0].ser_lower+a_d->info[0].ser_lower;
            _XMP_align_local_idx((long long int)(gj), &a_idx0, a_d, 0, &a_rank0);
            for(k=0; k<_XMP_get_execution_nodes()->comm_size; k++){
               b_offset[k] = 0;
            }
            for(i=x_d->info[1].local_lower; i<=x_d->info[1].local_upper; i++){
               int gi = l2g(x_d, 1, i)-x_d->info[1].ser_lower+b_d->info[1].ser_lower;
               _XMP_align_local_idx((long long int)(gi), &b_idx1, b_d, 1, &b_rank1);
               char *x_p = (char*)(x_d->array_addr_p)+(j*x_alloc_size[1]+i)*x_d->type_size;
               memset(x_p, 0, x_d->type_size);
               for(k=0; k<a_d->info[1].ser_size; k++){
                  _XMP_align_local_idx((long long int)(k+a_d->info[1].ser_lower), &a_idx1, a_d, 1, &a_rank1);
                  _XMP_align_local_idx((long long int)(k+b_d->info[0].ser_lower), &b_idx0, b_d, 0, &b_rank0);
                  char *a_p = (char*)(a_buf+a_recv_pos[a2e[g2p_array(a_d,gj,k+a_d->info[1].ser_lower)]]
                                      +(a_idx1+a_offset[a2e[g2p_array(a_d,gj,k+a_d->info[1].ser_lower)]])*a_d->type_size);
                  char *b_p = (char*)(b_buf+b_recv_pos[b2e[g2p_array(b_d,k+b_d->info[0].ser_lower,gi)]]
                                      +(b_offset[b2e[g2p_array(b_d,k+b_d->info[0].ser_lower,gi)]])*b_d->type_size);
#ifdef DEBUG
                  if(i==x_d->info[0].local_lower+7 && j==x_d->info[1].local_lower+7 &&
                     _XMP_get_execution_nodes()->comm_rank == 5){
                     printf("%4d x %4d: %d + %d, %d\n", *(int*)a_p, *(int*)b_p,
                            a_idx1, g2p_array(a_d,gj,k+a_d->info[1].ser_lower),
                            a_offset[a2e[g2p_array(a_d,gj,k+a_d->info[1].ser_lower)]]);
                  }
#endif
                  b_offset[b2e[g2p_array(b_d,k+b_d->info[0].ser_lower,gi)]]++;
                  var_mul(x_d, x_p, a_p, b_p);
               }
            }
            for(k=0; k<proc_size(a_d, 1); k++){
               a_offset[a2e[proc_rank(a_d, a_rank0, k)]] += a_recv_size[k];
            }
         }
      }
   }
      
#ifdef DEBUG
   fflush(stdout);
   MPI_Barrier(*(MPI_Comm*)(_XMP_get_execution_nodes()->comm));
   show_array(x_d, NULL);
#endif

   /* duplicate */
   array_duplicate(x_d, send_req, recv_req);

   /* free */
   _XMP_free(e2e);
   _XMP_free(e2x);
   _XMP_free(e2a);
   _XMP_free(e2b);
   _XMP_free(x2e);
   _XMP_free(a2e);
   _XMP_free(b2e);
   _XMP_free(send_req);
   _XMP_free(recv_req);
   if(send_buf) _XMP_free(send_buf);
   if(a_buf) _XMP_free(a_buf);
   if(b_buf) _XMP_free(b_buf);
   _XMP_free(send_pos);
   _XMP_free(send_size);
   _XMP_free(a_recv_pos);
   _XMP_free(a_recv_size);
   _XMP_free(b_recv_pos);
   _XMP_free(b_recv_size);
}


static void xmp_matmul_allgather(_XMP_array_t *x_d, _XMP_array_t *a_d, _XMP_array_t *b_d, int dist_dim)
{
   MPI_Comm *exec_comm;
   char *send_buf=NULL, *recv_buf=NULL;
   int regular=0;
   int type_size=x_d->type_size;
   int x_alloc_size[2];
   int a_alloc_size[2];
   int b_alloc_size[2];
   int i, j, k, l;
   
   exec_comm = (MPI_Comm*)(_XMP_get_execution_nodes()->comm);
   
#ifdef DEBUG
   show_array(a_d, NULL);
   show_array(b_d, NULL);
#endif
   
   /* allocate check */
   if(x_d->is_allocated){
      x_alloc_size[0] = x_d->info[0].alloc_size;
      x_alloc_size[1] = x_d->info[1].alloc_size;
   } else {
      x_alloc_size[0] = 0;
      x_alloc_size[1] = 0;
   }
   if(a_d->is_allocated){
      a_alloc_size[0] = a_d->info[0].alloc_size;
      a_alloc_size[1] = a_d->info[1].alloc_size;
   } else {
      a_alloc_size[0] = 0;
      a_alloc_size[1] = 0;
   }
   if(b_d->is_allocated){
      b_alloc_size[0] = b_d->info[0].alloc_size;
      b_alloc_size[1] = b_d->info[1].alloc_size;
   } else {
      b_alloc_size[0] = 0;
      b_alloc_size[1] = 0;
   }

   if(xmpf_running){           /* Fortran */
      if(dist_dim == 0){
         int recv_count[_XMP_get_execution_nodes()->comm_size];
         int recv_size[_XMP_get_execution_nodes()->comm_size];
         int recv_offset[_XMP_get_execution_nodes()->comm_size];
         int dim0_size = b_d->info[0].local_upper-b_d->info[0].local_lower+1;
         int dim1_size = b_d->info[1].local_upper-b_d->info[1].local_lower+1;
         if(b_d->info[dist_dim].align_manner == _XMP_N_ALIGN_BLOCK){
            if(b_d->info[dist_dim].ser_size%_XMP_get_execution_nodes()->comm_size == 0){
               regular = b_d->info[dist_dim].ser_size/_XMP_get_execution_nodes()->comm_size;
            }
         } else {               /* align GBLOCK */
            _XMP_template_chunk_t *chunk = &(b_d->align_template->chunk[b_d->info[dist_dim].align_template_index]);
            regular = chunk->mapping_array[1] - chunk->mapping_array[0];
            for(i=1; i<chunk->onto_nodes_info->size; i++){
               if(chunk->mapping_array[i+1] - chunk->mapping_array[i] != regular){
                  regular = 0;
                  break;
               }
            }
         }

         /* transpose & pack */
         send_buf = (char*)_XMP_alloc(dim0_size*dim1_size*type_size);
         char *dst_p = send_buf;
         char *src_p = (char*)(b_d->array_addr_p);
         for(i=b_d->info[0].local_lower; i<=b_d->info[0].local_upper; i++){
            for(j=b_d->info[1].local_lower; j<=b_d->info[1].local_upper; j++){
               memcpy(dst_p+(dim1_size*(i-b_d->info[0].local_lower)+j-b_d->info[1].local_lower)*type_size,
                      src_p+(b_alloc_size[0]*j+i)*type_size, type_size);
            }
         }
#ifdef DEBUG
         show_array_ij((int*)send_buf, dim1_size, dim0_size);
#endif
         recv_buf = (char*)_XMP_alloc(b_d->info[0].ser_size*b_d->info[1].ser_size*type_size);

         /* communication */
         if(regular){
            int count=regular*dim1_size*type_size;
            MPI_Allgather(send_buf, count, MPI_BYTE, recv_buf, count, MPI_BYTE, *exec_comm);
         } else {               /* not regular */
            int send_count;
            if(b_d->info[dist_dim].align_manner == _XMP_N_ALIGN_BLOCK){
               int w=b_d->info[dist_dim].ser_size/_XMP_get_execution_nodes()->comm_size+1;
               int w_all=b_d->info[dist_dim].ser_size;
               recv_count[0] = w;
               recv_size[0] = w*dim1_size*type_size;
               recv_offset[0] = 0;
               for(i=1; i<_XMP_get_execution_nodes()->comm_size; i++){
                  recv_count[i] = (w_all > w)? w: w_all;
                  recv_size[i] = recv_count[i]*dim1_size*type_size;
                  recv_offset[i] = recv_offset[i-1]+recv_size[i-1];
                  w_all -= recv_count[i];
               }
            } else {            /* align GBLOCK */
               _XMP_template_chunk_t *chunk = &(b_d->align_template->chunk[b_d->info[dist_dim].align_template_index]);
               recv_count[0] = chunk->mapping_array[1]-chunk->mapping_array[0];
               recv_size[0] = recv_count[0]*dim1_size*type_size;
               recv_offset[0] = 0;
               for(i=1; i<_XMP_get_execution_nodes()->comm_size; i++){
                  recv_count[i] = chunk->mapping_array[i+1]-chunk->mapping_array[i];
                  recv_size[i] = recv_count[i]*dim1_size*type_size;
                  recv_offset[i] = recv_offset[i-1]+recv_size[i-1];
               }
            }
            send_count = recv_count[_XMP_get_execution_nodes()->comm_rank];
#ifdef DEBUG
            for(k=0; k<_XMP_get_execution_nodes()->comm_size; k++){
               if(k == _XMP_get_execution_nodes()->comm_rank){
                  printf(" rank%d : send_size = %d: recv_size = ", k, send_count*dim1_size*type_size);
                  for(j=0; j<_XMP_get_execution_nodes()->comm_size; j++){
                     printf("%2d ", recv_size[j]);
                  }
                  printf("\n");
               }
               fflush(stdout);
               MPI_Barrier(*(MPI_Comm*)(_XMP_get_execution_nodes()->comm));
            }
#endif
            MPI_Allgatherv(send_buf, send_count*dim1_size*type_size, MPI_BYTE,
                           recv_buf, recv_size, recv_offset, MPI_BYTE, *exec_comm);
         }
         /* matmul */
#ifdef DEBUG
         show_array_ij((int*)recv_buf, b_d->info[1].ser_size, b_d->info[0].ser_size);
#endif
         memset((char*)x_d->array_addr_p, 0, x_alloc_size[0]*x_alloc_size[1]*type_size);
         for(k=0; k<b_d->info[0].ser_size; k++){
            for(j=x_d->info[1].local_lower; j<=x_d->info[1].local_upper; j++){
               int bj=j-x_d->info[1].local_lower;
               for(i=x_d->info[0].local_lower; i<=x_d->info[0].local_upper; i++){
                  int ai=i-x_d->info[0].local_lower+a_d->info[0].local_lower;
                  char *x_p = ((char*)x_d->array_addr_p+(j*x_alloc_size[0]+i)*x_d->type_size);
                  char *a_p = ((char*)a_d->array_addr_p+(a_alloc_size[0]*k+ai)*type_size);
                  char *b_p = (recv_buf+(b_d->info[1].ser_size*k+bj)*type_size);
                  var_mul(x_d, x_p, a_p, b_p);
/*                   *x_p += *a_p * *b_p; */
/* #ifdef DEBUG */
/*                   if(i==x_d->info[0].local_lower+0 && j==x_d->info[1].local_lower+1 && */
/*                      _XMP_get_execution_nodes()->comm_rank == 0){ */
/*                      printf("%4d x %4d\n", *a_p, *b_p); */
/*                   } */
/* #endif */
               }
            }
         }
         
      } else {                  /* dist_dim == 1 */
         int recv_count[_XMP_get_execution_nodes()->comm_size];
         int recv_size[_XMP_get_execution_nodes()->comm_size];
         int recv_offset[_XMP_get_execution_nodes()->comm_size];
         int dim0_size = a_d->info[0].local_upper-a_d->info[0].local_lower+1;
         int dim1_size = a_d->info[1].local_upper-a_d->info[1].local_lower+1;
         if(a_d->info[dist_dim].align_manner == _XMP_N_ALIGN_BLOCK){
            if(a_d->info[dist_dim].ser_size%_XMP_get_execution_nodes()->comm_size == 0){
               regular = a_d->info[dist_dim].ser_size/_XMP_get_execution_nodes()->comm_size;
            }
         } else {               /* align GBLOCK */
            _XMP_template_chunk_t *chunk = &(a_d->align_template->chunk[a_d->info[dist_dim].align_template_index]);
            regular = chunk->mapping_array[1] - chunk->mapping_array[0];
            for(i=1; i<chunk->onto_nodes_info->size; i++){
               if(chunk->mapping_array[i+1] - chunk->mapping_array[i] != regular){
                  regular = 0;
                  break;
               }
            }
         }

         /* pack */
         if(a_d->info[1].shadow_size_lo == 0 && a_d->info[1].shadow_size_hi == 0){
            send_buf = (char*)(a_d->array_addr_p);
         } else {
            send_buf = (char*)_XMP_alloc(dim0_size*dim1_size*type_size);
            char *dst_p = send_buf;
            char *src_p = (char*)(a_d->array_addr_p);
            for(i=a_d->info[1].local_lower; i<=a_d->info[1].local_upper; i++){
               memcpy(dst_p+(dim0_size*(i-a_d->info[1].local_lower))*type_size,
                      src_p+(a_alloc_size[0]*i)*type_size, type_size*dim0_size);
            }
         }
#ifdef DEBUG
         show_array_ij((int*)send_buf, dim0_size, dim1_size);
#endif
         recv_buf = (char*)_XMP_alloc(a_d->info[0].ser_size*a_d->info[1].ser_size*type_size);

         /* communication */
         if(regular){
            int count=regular*dim0_size*type_size;
            MPI_Allgather(send_buf, count, MPI_BYTE, recv_buf, count, MPI_BYTE, *exec_comm);
         } else {               /* not regular */
            int send_count;
            if(a_d->info[dist_dim].align_manner == _XMP_N_ALIGN_BLOCK){
               int w=a_d->info[dist_dim].ser_size/_XMP_get_execution_nodes()->comm_size+1;
               int w_all=a_d->info[dist_dim].ser_size;
               recv_count[0] = w;
               recv_size[0] = w*dim0_size*type_size;
               recv_offset[0] = 0;
               for(i=1; i<_XMP_get_execution_nodes()->comm_size; i++){
                  recv_count[i] = (w_all > w)? w: w_all;
                  recv_size[i] = recv_count[i]*dim0_size*type_size;
                  recv_offset[i] = recv_offset[i-1]+recv_size[i-1];
                  w_all -= recv_count[i];
               }
            } else {            /* align GBLOCK */
               _XMP_template_chunk_t *chunk = &(a_d->align_template->chunk[a_d->info[dist_dim].align_template_index]);
               recv_count[0] = chunk->mapping_array[1]-chunk->mapping_array[0];
               recv_size[0] = recv_count[0]*dim0_size*type_size;
               recv_offset[0] = 0;
               for(i=1; i<_XMP_get_execution_nodes()->comm_size; i++){
                  recv_count[i] = chunk->mapping_array[i+1]-chunk->mapping_array[i];
                  recv_size[i] = recv_count[i]*dim0_size*type_size;
                  recv_offset[i] = recv_offset[i-1]+recv_size[i-1];
               }
            }
            send_count = recv_count[_XMP_get_execution_nodes()->comm_rank];
#ifdef DEBUG
            for(k=0; k<_XMP_get_execution_nodes()->comm_size; k++){
               if(k == _XMP_get_execution_nodes()->comm_rank){
                  printf(" rank%d : send_size = %d: recv_size = ", k, send_count*dim0_size*type_size);
                  for(j=0; j<_XMP_get_execution_nodes()->comm_size; j++){
                     printf("%2d ", recv_size[j]);
                  }
                  printf("\n");
               }
               fflush(stdout);
               MPI_Barrier(*(MPI_Comm*)(_XMP_get_execution_nodes()->comm));
            }
#endif
            MPI_Allgatherv(send_buf, send_count*dim0_size*type_size, MPI_BYTE,
                           recv_buf, recv_size, recv_offset, MPI_BYTE, *exec_comm);
         }
         /* matmul */
#ifdef DEBUG
         show_array_ij((int*)recv_buf, a_d->info[0].ser_size, a_d->info[1].ser_size);
#endif
         memset((char*)x_d->array_addr_p, 0, x_alloc_size[0]*x_alloc_size[1]*type_size);
         for(k=0; k<b_d->info[0].ser_size; k++){
            for(j=x_d->info[1].local_lower; j<=x_d->info[1].local_upper; j++){
               int bj=j-x_d->info[1].local_lower;
               for(i=x_d->info[0].local_lower; i<=x_d->info[0].local_upper; i++){
                  int ai=i-x_d->info[0].local_lower+a_d->info[0].local_lower;
                  char *x_p = ((char*)x_d->array_addr_p+(j*x_alloc_size[0]+i)*x_d->type_size);
                  char *a_p = (recv_buf+(a_d->info[0].ser_size*k+ai)*type_size);
                  char *b_p = ((char*)b_d->array_addr_p+(b_alloc_size[0]*bj+k+b_d->info[0].local_lower)*type_size);
                  var_mul(x_d, x_p, a_p, b_p);
/*                   *x_p += *a_p * *b_p; */
/* #ifdef DEBUG */
/*                   if(i==x_d->info[0].local_lower+0 && j==x_d->info[1].local_lower+1 && */
/*                      _XMP_get_execution_nodes()->comm_rank == 0){ */
/*                      printf("%4d x %4d\n", *a_p, *b_p); */
/*                   } */
/* #endif */
               }
            }
         }
      }
      
   } else {                     /* C */
      if(dist_dim == 0){
         int recv_count[_XMP_get_execution_nodes()->comm_size];
         int recv_size[_XMP_get_execution_nodes()->comm_size];
         int recv_offset[_XMP_get_execution_nodes()->comm_size];
         int dim0_size = b_d->info[0].local_upper-b_d->info[0].local_lower+1;
         int dim1_size = b_d->info[1].local_upper-b_d->info[1].local_lower+1;
         if(b_d->info[dist_dim].align_manner == _XMP_N_ALIGN_BLOCK){
            if(b_d->info[dist_dim].ser_size%_XMP_get_execution_nodes()->comm_size == 0){
               regular = b_d->info[dist_dim].ser_size/_XMP_get_execution_nodes()->comm_size;
            }
         } else {               /* align GBLOCK */
            _XMP_template_chunk_t *chunk = &(b_d->align_template->chunk[b_d->info[dist_dim].align_template_index]);
            regular = chunk->mapping_array[1] - chunk->mapping_array[0];
            for(i=1; i<chunk->onto_nodes_info->size; i++){
               if(chunk->mapping_array[i+1] - chunk->mapping_array[i] != regular){
                  regular = 0;
                  break;
               }
            }
         }

         /* transpose & pack */
         send_buf = (char*)_XMP_alloc(dim0_size*dim1_size*type_size);
         char *dst_p = send_buf;
         char *src_p = (char*)(b_d->array_addr_p);
         for(i=b_d->info[0].local_lower; i<=b_d->info[0].local_upper; i++){
            for(j=b_d->info[1].local_lower; j<=b_d->info[1].local_upper; j++){
               memcpy(dst_p+(dim0_size*(j-b_d->info[1].local_lower)+i-b_d->info[0].local_lower)*type_size,
                      src_p+(b_alloc_size[1]*i+j)*type_size, type_size);
            }
         }
#ifdef DEBUG
         show_array_ij((int*)send_buf, dim1_size, dim0_size);
#endif
         recv_buf = (char*)_XMP_alloc(b_d->info[0].ser_size*b_d->info[1].ser_size*type_size);

         /* communication */
         if(regular){
            int count=regular*dim1_size*type_size;
            MPI_Allgather(send_buf, count, MPI_BYTE, recv_buf, count, MPI_BYTE, *exec_comm);
         } else {               /* not regular */
            int send_count;
            if(b_d->info[dist_dim].align_manner == _XMP_N_ALIGN_BLOCK){
               int w=b_d->info[dist_dim].ser_size/_XMP_get_execution_nodes()->comm_size+1;
               int w_all=b_d->info[dist_dim].ser_size;
               recv_count[0] = w;
               recv_size[0] = w*dim1_size*type_size;
               recv_offset[0] = 0;
               for(i=1; i<_XMP_get_execution_nodes()->comm_size; i++){
                  recv_count[i] = (w_all > w)? w: w_all;
                  recv_size[i] = recv_count[i]*dim1_size*type_size;
                  recv_offset[i] = recv_offset[i-1]+recv_size[i-1];
                  w_all -= recv_count[i];
               }
            } else {            /* align GBLOCK */
               _XMP_template_chunk_t *chunk = &(b_d->align_template->chunk[b_d->info[dist_dim].align_template_index]);
               recv_count[0] = chunk->mapping_array[1]-chunk->mapping_array[0];
               recv_size[0] = recv_count[0]*dim1_size*type_size;
               recv_offset[0] = 0;
               for(i=1; i<_XMP_get_execution_nodes()->comm_size; i++){
                  recv_count[i] = chunk->mapping_array[i+1]-chunk->mapping_array[i];
                  recv_size[i] = recv_count[i]*dim1_size*type_size;
                  recv_offset[i] = recv_offset[i-1]+recv_size[i-1];
               }
            }
            send_count = recv_count[_XMP_get_execution_nodes()->comm_rank];
#ifdef DEBUG
            for(k=0; k<_XMP_get_execution_nodes()->comm_size; k++){
               if(k == _XMP_get_execution_nodes()->comm_rank){
                  printf(" rank%d : send_size = %d: recv_size = ", k, send_count*dim1_size*type_size);
                  for(j=0; j<_XMP_get_execution_nodes()->comm_size; j++){
                     printf("%2d ", recv_size[j]);
                  }
                  printf("\n");
               }
               fflush(stdout);
               MPI_Barrier(*(MPI_Comm*)(_XMP_get_execution_nodes()->comm));
            }
#endif
            MPI_Allgatherv(send_buf, send_count*dim1_size*type_size, MPI_BYTE,
                           recv_buf, recv_size, recv_offset, MPI_BYTE, *exec_comm);
         }
         /* matmul */
#ifdef DEBUG
         show_array_ij((int*)recv_buf, b_d->info[1].ser_size, b_d->info[0].ser_size);
#endif
         if(regular){
            for(i=x_d->info[0].local_lower; i<=x_d->info[0].local_upper; i++){
               int ai=i-x_d->info[0].local_lower+a_d->info[0].local_lower;
               for(j=x_d->info[1].local_lower; j<=x_d->info[1].local_upper; j++){
                  int bj=j-x_d->info[1].local_lower;
                  char *x_p = ((char*)x_d->array_addr_p+(i*x_alloc_size[1]+j)*x_d->type_size);
                  memset(x_p, 0, type_size);
                  for(k=0; k<_XMP_get_execution_nodes()->comm_size; k++){
                     for(l=0; l<dim0_size; l++){
                        char *a_p = ((char*)a_d->array_addr_p+(ai*a_alloc_size[1]+k*dim0_size+l)*type_size);
                        char *b_p = (recv_buf+(dim0_size*dim1_size*k+bj*dim0_size+l)*type_size);
                        var_mul(x_d, x_p, a_p, b_p);
/*                         *x_p += *a_p * *b_p; */
/* #ifdef DEBUG */
/*                         if(i==x_d->info[0].local_lower+0 && j==x_d->info[1].local_lower+8 && */
/*                            _XMP_get_execution_nodes()->comm_rank == 0){ */
/*                            printf("%4d x %4d\n", *a_p, *b_p); */
/*                         } */
/* #endif */
                     }
                  }
               }
            }
         } else {               /* not regular */
            for(i=x_d->info[0].local_lower; i<=x_d->info[0].local_upper; i++){
               int ai=i-x_d->info[0].local_lower+a_d->info[0].local_lower;
               for(j=x_d->info[1].local_lower; j<=x_d->info[1].local_upper; j++){
                  int bj=j-x_d->info[1].local_lower;
                  char *x_p = ((char*)x_d->array_addr_p+(i*x_alloc_size[1]+j)*x_d->type_size);
                  *x_p = 0;
                  int kk = a_d->info[0].local_lower;
                  for(k=0; k<_XMP_get_execution_nodes()->comm_size; k++){
                     for(l=0; l<recv_count[k]; l++){
                        char *a_p = ((char*)a_d->array_addr_p+(ai*a_alloc_size[1]+kk)*type_size);
                        char *b_p = (recv_buf+recv_offset[k]+(bj*recv_count[k]+l)*type_size);
                        var_mul(x_d, x_p, a_p, b_p);
                        kk++;
/*                         *x_p += *a_p * *b_p; */
/* #ifdef DEBUG */
/*                         if(i==x_d->info[0].local_lower+0 && j==x_d->info[1].local_lower+8 && */
/*                            _XMP_get_execution_nodes()->comm_rank == 0){ */
/*                            printf("%4d x %4d\n", *a_p, *b_p); */
/*                         } */
/* #endif */
                     }
                  }
               }
            }
         }
         
      } else {                  /* dist_dim == 1 */
         int recv_count[_XMP_get_execution_nodes()->comm_size];
         int recv_size[_XMP_get_execution_nodes()->comm_size];
         int recv_offset[_XMP_get_execution_nodes()->comm_size];
         int dim0_size = a_d->info[0].local_upper-a_d->info[0].local_lower+1;
         int dim1_size = a_d->info[1].local_upper-a_d->info[1].local_lower+1;
         if(a_d->info[dist_dim].align_manner == _XMP_N_ALIGN_BLOCK){
            if(a_d->info[dist_dim].ser_size%_XMP_get_execution_nodes()->comm_size == 0){
               regular = a_d->info[dist_dim].ser_size/_XMP_get_execution_nodes()->comm_size;
            }
         } else {               /* align GBLOCK */
            _XMP_template_chunk_t *chunk = &(a_d->align_template->chunk[a_d->info[dist_dim].align_template_index]);
            regular = chunk->mapping_array[1] - chunk->mapping_array[0];
            for(i=1; i<chunk->onto_nodes_info->size; i++){
               if(chunk->mapping_array[i+1] - chunk->mapping_array[i] != regular){
                  regular = 0;
                  break;
               }
            }
         }

         /* pack */
         if(a_d->info[1].shadow_size_lo == 0 && a_d->info[1].shadow_size_hi == 0){
            send_buf = (char*)(a_d->array_addr_p);
         } else {
            send_buf = (char*)_XMP_alloc(dim0_size*dim1_size*type_size);
            char *dst_p = send_buf;
            char *src_p = (char*)(a_d->array_addr_p);
            for(i=a_d->info[0].local_lower; i<=a_d->info[0].local_upper; i++){
               memcpy(dst_p+(dim1_size*(i-a_d->info[0].local_lower))*type_size,
                      src_p+(a_alloc_size[1]*i)*type_size, type_size*dim1_size);
            }
         }
#ifdef DEBUG
         show_array_ij((int*)send_buf, dim1_size, dim0_size);
#endif
         recv_buf = (char*)_XMP_alloc(a_d->info[0].ser_size*a_d->info[1].ser_size*type_size);

         /* communication */
         if(regular){
            int count=regular*dim0_size*type_size;
            MPI_Allgather(send_buf, count, MPI_BYTE, recv_buf, count, MPI_BYTE, *exec_comm);
         } else {               /* not regular */
            int send_count;
            if(a_d->info[dist_dim].align_manner == _XMP_N_ALIGN_BLOCK){
               int w=a_d->info[dist_dim].ser_size/_XMP_get_execution_nodes()->comm_size+1;
               int w_all=a_d->info[dist_dim].ser_size;
               recv_count[0] = w;
               recv_size[0] = w*dim0_size*type_size;
               recv_offset[0] = 0;
               for(i=1; i<_XMP_get_execution_nodes()->comm_size; i++){
                  recv_count[i] = (w_all > w)? w: w_all;
                  recv_size[i] = recv_count[i]*dim0_size*type_size;
                  recv_offset[i] = recv_offset[i-1]+recv_size[i-1];
                  w_all -= recv_count[i];
               }
            } else {            /* align GBLOCK */
               _XMP_template_chunk_t *chunk = &(a_d->align_template->chunk[a_d->info[dist_dim].align_template_index]);
               recv_count[0] = chunk->mapping_array[1]-chunk->mapping_array[0];
               recv_size[0] = recv_count[0]*dim0_size*type_size;
               recv_offset[0] = 0;
               for(i=1; i<_XMP_get_execution_nodes()->comm_size; i++){
                  recv_count[i] = chunk->mapping_array[i+1]-chunk->mapping_array[i];
                  recv_size[i] = recv_count[i]*dim0_size*type_size;
                  recv_offset[i] = recv_offset[i-1]+recv_size[i-1];
               }
            }
            send_count = recv_count[_XMP_get_execution_nodes()->comm_rank];
#ifdef DEBUG
            for(k=0; k<_XMP_get_execution_nodes()->comm_size; k++){
               if(k == _XMP_get_execution_nodes()->comm_rank){
                  printf(" rank%d : send_size = %d: recv_size = ", k, send_count*dim0_size*type_size);
                  for(j=0; j<_XMP_get_execution_nodes()->comm_size; j++){
                     printf("%2d ", recv_size[j]);
                  }
                  printf("\n");
               }
               fflush(stdout);
               MPI_Barrier(*(MPI_Comm*)(_XMP_get_execution_nodes()->comm));
            }
#endif
            MPI_Allgatherv(send_buf, send_count*dim0_size*type_size, MPI_BYTE,
                           recv_buf, recv_size, recv_offset, MPI_BYTE, *exec_comm);
         }
         /* matmul */
#ifdef DEBUG
         show_array_ij((int*)recv_buf, a_d->info[0].ser_size, a_d->info[1].ser_size);
#endif
         if(regular){
            for(i=x_d->info[0].local_lower; i<=x_d->info[0].local_upper; i++){
               int ai=i-x_d->info[0].local_lower;
               for(j=x_d->info[1].local_lower; j<=x_d->info[1].local_upper; j++){
                  int bj=j-x_d->info[1].local_lower+b_d->info[1].local_lower;
                  char *x_p = ((char*)x_d->array_addr_p+(i*x_alloc_size[1]+j)*x_d->type_size);
                  memset(x_p, 0, type_size);
                  for(k=0; k<_XMP_get_execution_nodes()->comm_size; k++){
                     for(l=0; l<dim1_size; l++){
                        char *a_p = (recv_buf+(dim1_size*dim0_size*k+ai*dim1_size+l)*type_size);
                        char *b_p = ((char*)b_d->array_addr_p+(b_alloc_size[1]*(dim1_size*k+l)+bj)*type_size);
                        var_mul(x_d, x_p, a_p, b_p);
/*                         *x_p += *a_p * *b_p; */
/* #ifdef DEBUG */
/*                         if(i==x_d->info[0].local_lower+0 && j==x_d->info[1].local_lower+0 && */
/*                            _XMP_get_execution_nodes()->comm_rank == 0){ */
/*                            printf("%4d x %4d\n", *a_p, *b_p); */
/*                         } */
/* #endif */
                     }
                  }
               }
            }
         } else {               /* not regular */
            for(i=x_d->info[0].local_lower; i<=x_d->info[0].local_upper; i++){
               int ai=i-x_d->info[0].local_lower;
               for(j=x_d->info[1].local_lower; j<=x_d->info[1].local_upper; j++){
                  int bj=j-x_d->info[1].local_lower+b_d->info[0].local_lower;
                  char *x_p = ((char*)x_d->array_addr_p+(i*x_alloc_size[1]+j)*x_d->type_size);
                  *x_p = 0;
                  int kk=b_d->info[0].local_lower;
                  for(k=0; k<_XMP_get_execution_nodes()->comm_size; k++){
                     for(l=0; l<recv_count[k]; l++){
                        char *a_p = (recv_buf+recv_offset[k]+(recv_count[k]*ai+l)*type_size);
                        char *b_p = ((char*)b_d->array_addr_p+(b_alloc_size[1]*kk+bj)*type_size);
                        var_mul(x_d, x_p, a_p, b_p);
                        kk++;
/*                         *x_p += *a_p * *b_p; */
/* #ifdef DEBUG */
/*                         if(i==x_d->info[0].local_lower+0 && j==x_d->info[1].local_lower+0 && */
/*                            _XMP_get_execution_nodes()->comm_rank == 0){ */
/*                            printf("%4d x %4d\n", *a_p, *b_p); */
/*                         } */
/* #endif */
                     }
                  }
               }
            }
         }
      }
   }
   
#ifdef DEBUG
   fflush(stdout);
   MPI_Barrier(*(MPI_Comm*)(_XMP_get_execution_nodes()->comm));
   show_array(x_d, NULL);
#endif

   if(send_buf && send_buf != a_d->array_addr_p && send_buf != b_d->array_addr_p) _XMP_free(send_buf);
   if(recv_buf) _XMP_free(recv_buf);
}

static void xmp_matmul_blockf(_XMP_array_t *x_d, _XMP_array_t *a_d, _XMP_array_t *b_d)
{
   MPI_Comm *exec_comm;
   MPI_Comm a_comm;
   MPI_Comm b_comm;
   char *send_buf=NULL, *a_send_buf, *b_send_buf, *a_recv_buf=NULL, *b_recv_buf = NULL;
   int dim0_size, dim1_size;
   int a_count[_XMP_get_execution_nodes()->comm_size];
   int a_size[_XMP_get_execution_nodes()->comm_size];
   int a_offset[_XMP_get_execution_nodes()->comm_size];
   int b_count[_XMP_get_execution_nodes()->comm_size];
   int b_size[_XMP_get_execution_nodes()->comm_size];
   int b_offset[_XMP_get_execution_nodes()->comm_size];
   int regular=0;
   int type_size=x_d->type_size;
   int x_alloc_size[2];
   int a_alloc_size[2];
   int b_alloc_size[2];
   int i, j, k;
   
   exec_comm = (MPI_Comm*)(_XMP_get_execution_nodes()->comm);
   
#ifdef DEBUG
   show_all(a_d);
   show_array(a_d, NULL);
#endif

   /* allocate check */
   if(x_d->is_allocated){
      x_alloc_size[0] = x_d->info[0].alloc_size;
      x_alloc_size[1] = x_d->info[1].alloc_size;
   } else {
      x_alloc_size[0] = 0;
      x_alloc_size[1] = 0;
   }
   if(a_d->is_allocated){
      a_alloc_size[0] = a_d->info[0].alloc_size;
      a_alloc_size[1] = a_d->info[1].alloc_size;
   } else {
      a_alloc_size[0] = 0;
      a_alloc_size[1] = 0;
   }
   if(b_d->is_allocated){
      b_alloc_size[0] = b_d->info[0].alloc_size;
      b_alloc_size[1] = b_d->info[1].alloc_size;
   } else {
      b_alloc_size[0] = 0;
      b_alloc_size[1] = 0;
   }

   /* send buffer allocate */
   dim0_size=(x_alloc_size[0] > a_alloc_size[0])? x_alloc_size[0]: a_alloc_size[0];
   dim0_size=(dim0_size > b_alloc_size[0])? dim0_size: b_alloc_size[0];
   dim1_size=(x_alloc_size[1] > a_alloc_size[1])? x_alloc_size[1]: a_alloc_size[1];
   dim1_size=(dim1_size > b_alloc_size[1])? dim1_size: b_alloc_size[1];
   send_buf = (char*)_XMP_alloc(dim0_size*dim1_size*type_size);
   
   /* a gather */
   /* TODO: use send/recv delete MPI_Comm_split */
   MPI_Comm_split(*exec_comm,
                  a_d->align_template->chunk[a_d->info[0].align_template_index].onto_nodes_info->rank,
                  _XMP_get_execution_nodes()->comm_rank,
                  &a_comm);
   
   dim0_size = a_d->info[0].local_upper-a_d->info[0].local_lower+1;
   dim1_size = a_d->info[1].local_upper-a_d->info[1].local_lower+1;
   if(a_d->info[1].align_manner == _XMP_N_ALIGN_BLOCK){
      int ap_size = a_d->align_template->chunk[a_d->info[1].align_template_index].onto_nodes_info->size;
      if(a_d->info[1].ser_size%ap_size == 0){
         regular = a_d->info[1].ser_size/ap_size;
      }
   } else {               /* align GBLOCK */
      _XMP_template_chunk_t *chunk = &(a_d->align_template->chunk[a_d->info[1].align_template_index]);
      regular = chunk->mapping_array[1] - chunk->mapping_array[0];
      for(i=1; i<chunk->onto_nodes_info->size; i++){
         if(chunk->mapping_array[i+1] - chunk->mapping_array[i] != regular){
            regular = 0;
            break;
         }
      }
   }
   
   /* pack */
   if(a_d->info[0].shadow_size_lo == 0 && a_d->info[0].shadow_size_hi == 0){
      a_send_buf = (char*)a_d->array_addr_p+a_alloc_size[0]*a_d->info[1].local_lower;
   } else {
      a_send_buf = send_buf;
      char *dst_p = send_buf;
      char *src_p = (char*)(a_d->array_addr_p);
      for(i=a_d->info[1].local_lower; i<=a_d->info[1].local_upper; i++){
         memcpy(dst_p+(dim0_size*(i-a_d->info[1].local_lower))*type_size,
                src_p+(a_alloc_size[0]*i+a_d->info[0].local_lower)*type_size, type_size*dim0_size);
      }
   }
   a_recv_buf = (char*)_XMP_alloc(dim0_size*a_d->info[1].ser_size*type_size);
   
   /* communication */
   if(regular){
      int count=dim0_size*regular*type_size;
      /* TODO: replace MPI_Allgather to send/recv */
      MPI_Allgather(a_send_buf, count, MPI_BYTE, a_recv_buf, count, MPI_BYTE, a_comm);
   } else {               /* not regular */
      _XMP_template_chunk_t *chunk = &(a_d->align_template->chunk[a_d->info[1].align_template_index]);
      int send_count;
      if(a_d->info[1].align_manner == _XMP_N_ALIGN_BLOCK){
         int ap_size = chunk->onto_nodes_info->size;
         int w=a_d->info[1].ser_size/ap_size+1;
         int w_all=a_d->info[1].ser_size-w;
         a_count[0] = w;
         a_size[0] = w*dim0_size*type_size;
         a_offset[0] = 0;
         for(i=1; i<chunk->onto_nodes_info->size; i++){
            a_count[i] = (w_all > w)? w: w_all;
            a_size[i] = a_count[i]*dim0_size*type_size;
            a_offset[i] = a_offset[i-1]+a_size[i-1];
            w_all -= a_count[i];
         }
      } else {            /* align GBLOCK */
         a_count[0] = chunk->mapping_array[1]-chunk->mapping_array[0];
         a_size[0] = a_count[0]*dim0_size*type_size;
         a_offset[0] = 0;
         for(i=1; i<chunk->onto_nodes_info->size; i++){
            a_count[i] = chunk->mapping_array[i+1]-chunk->mapping_array[i];
            a_size[i] = a_count[i]*dim0_size*type_size;
            a_offset[i] = a_offset[i-1]+a_size[i-1];
         }
      }
      send_count = a_count[chunk->onto_nodes_info->rank];
#ifdef DEBUG
      for(k=0; k<_XMP_get_execution_nodes()->comm_size; k++){
         if(k == _XMP_get_execution_nodes()->comm_rank){
            printf(" rank%d : send_size = %d: recv_size = ", k, send_count*dim0_size*type_size);
            for(j=0; j<_XMP_get_execution_nodes()->comm_size; j++){
               printf("%2d ", a_size[j]);
            }
            printf("\n");
         }
         fflush(stdout);
         MPI_Barrier(*(MPI_Comm*)(_XMP_get_execution_nodes()->comm));
      }
#endif
      /* TODO: replace MPI_Allgatherv to send/recv */
      MPI_Allgatherv(a_send_buf, send_count*dim0_size*type_size, MPI_BYTE,
                     a_recv_buf, a_size, a_offset, MPI_BYTE, a_comm);
   }
#ifdef DEBUG
   show_array_ij((int*)a_recv_buf, dim0_size, a_d->info[1].ser_size);
#endif
   
#ifdef DEBUG
   show_all(b_d);
   show_array(b_d, NULL);
#endif
   /* b gather */
   /* TODO: use send/recv delete MPI_Comm_split */
   MPI_Comm_split(*exec_comm,
                  b_d->align_template->chunk[b_d->info[1].align_template_index].onto_nodes_info->rank,
                  _XMP_get_execution_nodes()->comm_rank,
                  &b_comm);

   dim0_size = b_d->info[0].local_upper-b_d->info[0].local_lower+1;
   dim1_size = b_d->info[1].local_upper-b_d->info[1].local_lower+1;
   if(b_d->info[0].align_manner == _XMP_N_ALIGN_BLOCK){
      int bp_size = b_d->align_template->chunk[b_d->info[0].align_template_index].onto_nodes_info->size;
      if(b_d->info[0].ser_size%bp_size == 0){
         regular = b_d->info[0].ser_size/bp_size;
      }
   } else {               /* align GBLOCK */
      _XMP_template_chunk_t *chunk = &(b_d->align_template->chunk[b_d->info[0].align_template_index]);
      regular = chunk->mapping_array[1] - chunk->mapping_array[0];
      for(i=1; i<chunk->onto_nodes_info->size; i++){
         if(chunk->mapping_array[i+1] - chunk->mapping_array[i] != regular){
            regular = 0;
            break;
         }
      }
   }

   /* transpose & pack */
   b_send_buf = send_buf;
   char *dst_p = b_send_buf;
   char *src_p = (char*)(b_d->array_addr_p);
   for(i=b_d->info[0].local_lower; i<=b_d->info[0].local_upper; i++){
      for(j=b_d->info[1].local_lower; j<=b_d->info[1].local_upper; j++){
         memcpy(dst_p+(dim1_size*(i-b_d->info[0].local_lower)+j-b_d->info[1].local_lower)*type_size,
                src_p+(b_alloc_size[0]*j+i)*type_size, type_size);
      }
   }
   b_recv_buf = (char*)_XMP_alloc(b_d->info[0].ser_size*dim1_size*type_size);

   /* communication */
   if(regular){
      int count=regular*dim1_size*type_size;
      /* TODO: replace MPI_Allgather to send/recv */
      MPI_Allgather(b_send_buf, count, MPI_BYTE, b_recv_buf, count, MPI_BYTE, b_comm);
   } else {               /* not regular */
      _XMP_template_chunk_t *chunk = &(b_d->align_template->chunk[b_d->info[0].align_template_index]);
      int send_count;
      if(b_d->info[0].align_manner == _XMP_N_ALIGN_BLOCK){
         int bp_size = chunk->onto_nodes_info->size;
         int w=b_d->info[0].ser_size/bp_size+1;
         int w_all=b_d->info[0].ser_size-w;
         b_count[0] = w;
         b_size[0] = w*dim1_size*type_size;
         b_offset[0] = 0;
         for(i=1; i<chunk->onto_nodes_info->size; i++){
            b_count[i] = (w_all > w)? w: w_all;
            b_size[i] = b_count[i]*dim1_size*type_size;
            b_offset[i] = b_offset[i-1]+b_size[i-1];
            w_all -= b_count[i];
         }
      } else {            /* align GBLOCK */
         b_count[0] = chunk->mapping_array[1]-chunk->mapping_array[0];
         b_size[0] = b_count[0]*dim1_size*type_size;
         b_offset[0] = 0;
         for(i=1; i<chunk->onto_nodes_info->size; i++){
            b_count[i] = chunk->mapping_array[i+1]-chunk->mapping_array[i];
            b_size[i] = b_count[i]*dim1_size*type_size;
            b_offset[i] = b_offset[i-1]+b_size[i-1];
         }
      }
      send_count = b_count[chunk->onto_nodes_info->rank];
#ifdef DEBUG
      for(k=0; k<_XMP_get_execution_nodes()->comm_size; k++){
         if(k == _XMP_get_execution_nodes()->comm_rank){
            printf(" rank%d : send_size = %d: recv_size = ", k, send_count*dim1_size*type_size);
            for(j=0; j<_XMP_get_execution_nodes()->comm_size; j++){
               printf("%2d ", b_size[j]);
            }
            printf("\n");
         }
         fflush(stdout);
         MPI_Barrier(*(MPI_Comm*)(_XMP_get_execution_nodes()->comm));
      }
#endif
      /* TODO: replace MPI_Allgatherv to send/recv */
      MPI_Allgatherv(b_send_buf, send_count*dim1_size*type_size, MPI_BYTE,
                     b_recv_buf, b_size, b_offset, MPI_BYTE, b_comm);
   }
#ifdef DEBUG
   show_array_ij((int*)b_recv_buf, dim1_size, b_d->info[0].ser_size);
#endif

   /* matmul */
   /* TODO: X = A * BT -> DGEMM */
#ifdef OMNI_TARGET_CPU_KCOMPUTER
   dim0_size = x_d->info[0].local_upper - x_d->info[0].local_lower + 1;
   dim1_size = x_d->info[1].local_upper - x_d->info[1].local_lower + 1;
   k = a_d->info[1].ser_size;
   switch(x_d->type){
   /* case _XMP_N_TYPE_FLOAT: */
   /*    { */
   /*       char *dst_p = (char*)x_d->array_addr_p */
   /*          + (x_alloc_size[0]*x_d->info[1].local_lower+x_d->info[0].local_lower)*type_size; */
   /*       float alpha=1.0; */
   /*       float beta=0.0; */
   /*       int   ldc = x_alloc_size[0]; */
   /*       sgemm_("N", "T", &dim0_size, &dim1_size, &k, &alpha, (float*)a_recv_buf, &dim0_size, */
   /*              (float*)b_recv_buf, &dim1_size, &beta, (float*)dst_p, &ldc, 1, 1); */
   /*    } */
   /*    break; */
   case _XMP_N_TYPE_DOUBLE:
      {
         char *dst_p = (char*)x_d->array_addr_p
            + (x_alloc_size[0]*x_d->info[1].local_lower+x_d->info[0].local_lower)*type_size;
         double alpha=1.0;
         double beta=0.0;
         int   ldc = x_alloc_size[0];
         dgemm_("N", "T", &dim0_size, &dim1_size, &k, &alpha, (double*)a_recv_buf, &dim0_size,
                (double*)b_recv_buf, &dim1_size, &beta, (double*)dst_p, &ldc, 1, 1);
      }
      break;
   default:
      memset((char*)x_d->array_addr_p, 0, x_alloc_size[0]*x_alloc_size[1]*type_size);
      int iblk=128/type_size*8;
      int jblk=8;
      for(k=0; k<b_d->info[0].ser_size; k++){
         for(j=0; j<dim1_size; j+=jblk){
            for(i=0; i<dim0_size; i+=iblk){
               for(int jj=j; jj<j+jblk && jj<dim1_size; jj++){
                  char *b_p = b_recv_buf+(k*dim1_size+jj)*type_size;
                  for(int ii=i; ii<i+iblk && ii<dim0_size; ii++){
                     char *x_p = ((char*)x_d->array_addr_p+((jj+x_d->info[1].local_lower)*x_alloc_size[0]
                                                            +ii+x_d->info[0].local_lower)*type_size);
                     char *a_p = a_recv_buf+(k*dim0_size+ii)*type_size;
                     var_mul(x_d, x_p, a_p, b_p);
                  }
               }
            }
         }
      }
      break;
   }
#else
   dim0_size = x_d->info[0].local_upper-x_d->info[0].local_lower+1;
   dim1_size = x_d->info[1].local_upper-x_d->info[1].local_lower+1;
   memset((char*)x_d->array_addr_p, 0, x_alloc_size[0]*x_alloc_size[1]*type_size);
   int iblk=128/type_size*8;
   int jblk=8;
   for(k=0; k<b_d->info[0].ser_size; k++){
      for(j=0; j<dim1_size; j+=jblk){
         for(i=0; i<dim0_size; i+=iblk){
            for(int jj=j; jj<j+jblk && jj<dim1_size; jj++){
               char *b_p = b_recv_buf+(k*dim1_size+jj)*type_size;
               for(int ii=i; ii<i+iblk && ii<dim0_size; ii++){
                  char *x_p = ((char*)x_d->array_addr_p+((jj+x_d->info[1].local_lower)*x_alloc_size[0]
                                                         +ii+x_d->info[0].local_lower)*type_size);
                  char *a_p = a_recv_buf+(k*dim0_size+ii)*type_size;
                  var_mul(x_d, x_p, a_p, b_p);
               }
            }
         }
      }
   }
#endif
   
#ifdef DEBUG
   fflush(stdout);
   MPI_Barrier(*(MPI_Comm*)(_XMP_get_execution_nodes()->comm));
   show_array(x_d, NULL);
#endif

   /* TODO: use send/recv delete MPI_Comm_free */
   MPI_Comm_free(&a_comm);
   MPI_Comm_free(&b_comm);
   if(send_buf) _XMP_free(send_buf);
   if(a_recv_buf) _XMP_free(a_recv_buf);
   if(b_recv_buf) _XMP_free(b_recv_buf);
}


static void xmp_matmul_blockc(_XMP_array_t *x_d, _XMP_array_t *a_d, _XMP_array_t *b_d)
{
   MPI_Comm *exec_comm;
   MPI_Comm a_comm;
   MPI_Comm b_comm;
   char *send_buf=NULL, *a_send_buf, *b_send_buf, *a_recv_buf=NULL, *b_recv_buf = NULL;
   int dim0_size, dim1_size;
   int a_count[_XMP_get_execution_nodes()->comm_size];
   int a_size[_XMP_get_execution_nodes()->comm_size];
   int a_offset[_XMP_get_execution_nodes()->comm_size];
   int b_count[_XMP_get_execution_nodes()->comm_size];
   int b_size[_XMP_get_execution_nodes()->comm_size];
   int b_offset[_XMP_get_execution_nodes()->comm_size];
   int regular=0;
   int type_size=x_d->type_size;
   int x_alloc_size[2];
   int a_alloc_size[2];
   int b_alloc_size[2];
   int i, j, k;
   
   exec_comm = (MPI_Comm*)(_XMP_get_execution_nodes()->comm);
   
#ifdef DEBUG
   show_all(a_d);
   show_array(a_d, NULL);
#endif

   /* allocate check */
   if(x_d->is_allocated){
      x_alloc_size[0] = x_d->info[0].alloc_size;
      x_alloc_size[1] = x_d->info[1].alloc_size;
   } else {
      x_alloc_size[0] = 0;
      x_alloc_size[1] = 0;
   }
   if(a_d->is_allocated){
      a_alloc_size[0] = a_d->info[0].alloc_size;
      a_alloc_size[1] = a_d->info[1].alloc_size;
   } else {
      a_alloc_size[0] = 0;
      a_alloc_size[1] = 0;
   }
   if(b_d->is_allocated){
      b_alloc_size[0] = b_d->info[0].alloc_size;
      b_alloc_size[1] = b_d->info[1].alloc_size;
   } else {
      b_alloc_size[0] = 0;
      b_alloc_size[1] = 0;
   }

   /* send buffer allocate */
   dim0_size=(x_alloc_size[0] > a_alloc_size[0])? x_alloc_size[0]: a_alloc_size[0];
   dim0_size=(dim0_size > b_alloc_size[0])? dim0_size: b_alloc_size[0];
   dim1_size=(x_alloc_size[1] > a_alloc_size[1])? x_alloc_size[1]: a_alloc_size[1];
   dim1_size=(dim1_size > b_alloc_size[1])? dim1_size: b_alloc_size[1];
   send_buf = (char*)_XMP_alloc(dim0_size*dim1_size*type_size);

   /* a gather */
   /* TODO: use send/recv delete MPI_Comm_split */
   MPI_Comm_split(*exec_comm,
                  a_d->align_template->chunk[a_d->info[0].align_template_index].onto_nodes_info->rank,
                  _XMP_get_execution_nodes()->comm_rank,
                  &a_comm);

   dim0_size = a_d->info[0].local_upper-a_d->info[0].local_lower+1;
   dim1_size = a_d->info[1].local_upper-a_d->info[1].local_lower+1;
   if(a_d->info[1].align_manner == _XMP_N_ALIGN_BLOCK){
      int ap_size = a_d->align_template->chunk[a_d->info[1].align_template_index].onto_nodes_info->size;
      if(a_d->info[1].ser_size%ap_size == 0){
         regular = a_d->info[1].ser_size/ap_size;
      }
   } else {               /* align GBLOCK */
      _XMP_template_chunk_t *chunk = &(a_d->align_template->chunk[a_d->info[1].align_template_index]);
      regular = chunk->mapping_array[1] - chunk->mapping_array[0];
      for(i=1; i<chunk->onto_nodes_info->size; i++){
         if(chunk->mapping_array[i+1] - chunk->mapping_array[i] != regular){
            regular = 0;
            break;
         }
      }
   }
   
   /* transpose & pack */
   a_send_buf = send_buf;
   char *dst_p = a_send_buf;
   char *src_p = (char*)(a_d->array_addr_p);
   for(i=a_d->info[0].local_lower; i<=a_d->info[0].local_upper; i++){
      for(j=a_d->info[1].local_lower; j<=a_d->info[1].local_upper; j++){
         memcpy(dst_p+(dim0_size*(j-a_d->info[1].local_lower)+i-a_d->info[0].local_lower)*type_size,
                src_p+(a_alloc_size[1]*i+j)*type_size, type_size);
      }
   }
   a_recv_buf = (char*)_XMP_alloc(dim0_size*a_d->info[1].ser_size*type_size);
   
   /* communication */
   if(regular){
      int count=dim0_size*regular*type_size;
      /* TODO: replace MPI_Allgather to send/recv */
      MPI_Allgather(a_send_buf, count, MPI_BYTE, a_recv_buf, count, MPI_BYTE, a_comm);
   } else {               /* not regular */
      _XMP_template_chunk_t *chunk = &(a_d->align_template->chunk[a_d->info[1].align_template_index]);
      int send_count;
      if(a_d->info[1].align_manner == _XMP_N_ALIGN_BLOCK){
         int ap_size = chunk->onto_nodes_info->size;
         int w=a_d->info[1].ser_size/ap_size+1;
         int w_all=a_d->info[1].ser_size-w;
         a_count[0] = w;
         a_size[0] = w*dim0_size*type_size;
         a_offset[0] = 0;
         for(i=1; i<chunk->onto_nodes_info->size; i++){
            a_count[i] = (w_all > w)? w: w_all;
            a_size[i] = a_count[i]*dim0_size*type_size;
            a_offset[i] = a_offset[i-1]+a_size[i-1];
            w_all -= a_count[i];
         }
      } else {            /* align GBLOCK */
         a_count[0] = chunk->mapping_array[1]-chunk->mapping_array[0];
         a_size[0] = a_count[0]*dim0_size*type_size;
         a_offset[0] = 0;
         for(i=1; i<chunk->onto_nodes_info->size; i++){
            a_count[i] = chunk->mapping_array[i+1]-chunk->mapping_array[i];
            a_size[i] = a_count[i]*dim0_size*type_size;
            a_offset[i] = a_offset[i-1]+a_size[i-1];
         }
      }
      send_count = a_count[chunk->onto_nodes_info->rank];
#ifdef DEBUG
      for(k=0; k<_XMP_get_execution_nodes()->comm_size; k++){
         if(k == _XMP_get_execution_nodes()->comm_rank){
            printf(" rank%d : send_size = %d: recv_size = ", k, send_count*dim0_size*type_size);
            for(j=0; j<_XMP_get_execution_nodes()->comm_size; j++){
               printf("%2d ", a_size[j]);
            }
            printf("\n");
         }
         fflush(stdout);
         MPI_Barrier(*(MPI_Comm*)(_XMP_get_execution_nodes()->comm));
      }
#endif
      /* TODO: replace MPI_Allgatherv to send/recv */
      MPI_Allgatherv(a_send_buf, send_count*dim0_size*type_size, MPI_BYTE,
                     a_recv_buf, a_size, a_offset, MPI_BYTE, a_comm);
   }
#ifdef DEBUG
   show_array_ij((int*)a_recv_buf, a_d->info[1].ser_size, dim0_size);
#endif
   
#ifdef DEBUG
   show_all(b_d);
   show_array(b_d, NULL);
#endif
   /* b gather */
   /* TODO: use send/recv delete MPI_Comm_split */
   MPI_Comm_split(*exec_comm,
                  b_d->align_template->chunk[b_d->info[1].align_template_index].onto_nodes_info->rank,
                  _XMP_get_execution_nodes()->comm_rank,
                  &b_comm);

   dim0_size = b_d->info[0].local_upper-b_d->info[0].local_lower+1;
   dim1_size = b_d->info[1].local_upper-b_d->info[1].local_lower+1;
   if(b_d->info[0].align_manner == _XMP_N_ALIGN_BLOCK){
      int bp_size = b_d->align_template->chunk[b_d->info[0].align_template_index].onto_nodes_info->size;
      if(b_d->info[0].ser_size%bp_size == 0){
         regular = b_d->info[0].ser_size/bp_size;
      }
   } else {               /* align GBLOCK */
      _XMP_template_chunk_t *chunk = &(b_d->align_template->chunk[b_d->info[0].align_template_index]);
      regular = chunk->mapping_array[1] - chunk->mapping_array[0];
      for(i=1; i<chunk->onto_nodes_info->size; i++){
         if(chunk->mapping_array[i+1] - chunk->mapping_array[i] != regular){
            regular = 0;
            break;
         }
      }
   }

   /* pack */
   if(b_d->info[1].shadow_size_lo == 0 && b_d->info[1].shadow_size_hi == 0){
      b_send_buf = (char*)b_d->array_addr_p+b_alloc_size[1]*b_d->info[0].local_lower;
   } else {
      b_send_buf = send_buf;
      char *dst_p = b_send_buf;
      char *src_p = (char*)(b_d->array_addr_p);
      for(i=b_d->info[0].local_lower; i<=b_d->info[0].local_upper; i++){
         memcpy(dst_p+(dim1_size*(i-b_d->info[0].local_lower))*type_size,
                src_p+(b_alloc_size[1]*i+b_d->info[1].local_lower)*type_size, type_size*dim1_size);
      }
   }
   b_recv_buf = (char*)_XMP_alloc(b_d->info[0].ser_size*dim1_size*type_size);

   /* communication */
   if(regular){
      int count=regular*dim1_size*type_size;
      /* TODO: replace MPI_Allgather to send/recv */
      MPI_Allgather(b_send_buf, count, MPI_BYTE, b_recv_buf, count, MPI_BYTE, b_comm);
   } else {               /* not regular */
      _XMP_template_chunk_t *chunk = &(b_d->align_template->chunk[b_d->info[0].align_template_index]);
      int send_count;
      if(b_d->info[0].align_manner == _XMP_N_ALIGN_BLOCK){
         int bp_size = chunk->onto_nodes_info->size;
         int w=b_d->info[0].ser_size/bp_size+1;
         int w_all=b_d->info[0].ser_size-w;
         b_count[0] = w;
         b_size[0] = w*dim1_size*type_size;
         b_offset[0] = 0;
         for(i=1; i<chunk->onto_nodes_info->size; i++){
            b_count[i] = (w_all > w)? w: w_all;
            b_size[i] = b_count[i]*dim1_size*type_size;
            b_offset[i] = b_offset[i-1]+b_size[i-1];
            w_all -= b_count[i];
         }
      } else {            /* align GBLOCK */
         b_count[0] = chunk->mapping_array[1]-chunk->mapping_array[0];
         b_size[0] = b_count[0]*dim1_size*type_size;
         b_offset[0] = 0;
         for(i=1; i<chunk->onto_nodes_info->size; i++){
            b_count[i] = chunk->mapping_array[i+1]-chunk->mapping_array[i];
            b_size[i] = b_count[i]*dim1_size*type_size;
            b_offset[i] = b_offset[i-1]+b_size[i-1];
         }
      }
      send_count = b_count[chunk->onto_nodes_info->rank];
#ifdef DEBUG
      for(k=0; k<_XMP_get_execution_nodes()->comm_size; k++){
         if(k == _XMP_get_execution_nodes()->comm_rank){
            printf(" rank%d : send_size = %d: recv_size = ", k, send_count*dim1_size*type_size);
            for(j=0; j<_XMP_get_execution_nodes()->comm_size; j++){
               printf("%2d ", b_size[j]);
            }
            printf("\n");
         }
         fflush(stdout);
         MPI_Barrier(*(MPI_Comm*)(_XMP_get_execution_nodes()->comm));
      }
#endif
      /* TODO: replace MPI_Allgatherv to send/recv */
      MPI_Allgatherv(b_send_buf, send_count*dim1_size*type_size, MPI_BYTE,
                     b_recv_buf, b_size, b_offset, MPI_BYTE, b_comm);
   }
#ifdef DEBUG
   show_array_ij((int*)b_recv_buf, b_d->info[0].ser_size, dim1_size);
#endif

   /* matmul */
   /* TODO: X = AT * B -> DGEMM */
   dim0_size = x_d->info[0].local_upper-x_d->info[0].local_lower+1;
   dim1_size = x_d->info[1].local_upper-x_d->info[1].local_lower+1;
   memset((char*)x_d->array_addr_p, 0, x_alloc_size[0]*x_alloc_size[1]*type_size);
   int iblk=128/type_size*8;
   int jblk=8;
   for(k=0; k<b_d->info[0].ser_size; k++){
      for(i=0; i<dim0_size; i+=iblk){
         for(j=0; j<dim1_size; j+=jblk){
            for(int ii=i; ii<i+iblk && ii<dim0_size; ii++){
               char *a_p = a_recv_buf+(k*dim0_size+ii)*type_size;
               for(int jj=j; jj<j+jblk && jj<dim1_size; jj++){
                  char *x_p = ((char*)x_d->array_addr_p+((ii+x_d->info[0].local_lower)*x_alloc_size[1]
                                                         +jj+x_d->info[1].local_lower)*type_size);
                  char *b_p = b_recv_buf+(k*dim1_size+jj)*type_size;
#ifdef DEBUG
                  if(ii==x_d->info[0].local_lower+9 && jj==x_d->info[1].local_lower+7 &&
                     _XMP_get_execution_nodes()->comm_rank == 3){
                     printf("%4d x %4d: %d*%d+%d\n", *((int*)a_p), *((int*)b_p), k, dim0_size, ii);
                  }
#endif
                  var_mul(x_d, x_p, a_p, b_p);
               }
            }
         }
      }
   }
   
#ifdef DEBUG
   fflush(stdout);
   MPI_Barrier(*(MPI_Comm*)(_XMP_get_execution_nodes()->comm));
   show_array(x_d, NULL);
#endif

   /* TODO: use send/recv delete MPI_Comm_free */
   MPI_Comm_free(&a_comm);
   MPI_Comm_free(&b_comm);
   if(send_buf) _XMP_free(send_buf);
   if(a_recv_buf) _XMP_free(a_recv_buf);
   if(b_recv_buf) _XMP_free(b_recv_buf);
}


void xmp_matmul(void *x_p, void *a_p, void *b_p)
{
   _XMP_array_t *x_d;
   _XMP_array_t *a_d;
   _XMP_array_t *b_d;
   int same_nodes;
   int same_align;
   int duplicate;
   int dist_dim=0;
   int i;
   
   x_d = (_XMP_array_t*)x_p;
   a_d = (_XMP_array_t*)a_p;
   b_d = (_XMP_array_t*)b_p;

   
   /* error check */
   if(x_d->dim != 2 || a_d->dim != 2 || b_d->dim != 2){
      _XMP_fatal("xmp_matmul: argument dimension is not 2");
      return;
   }
   if(x_d->type != a_d->type || x_d->type != b_d->type){
      _XMP_fatal("xmp_matmul: argument type is not match");
      return;
   }
   if(!x_d->align_template->is_distributed ||
      !a_d->align_template->is_distributed ||
      !b_d->align_template->is_distributed){
      _XMP_fatal("xmp_matmul: argument is not distributed");
      return;
   }

   /* same nodes? */
   same_nodes = 1;
   if(_XMP_get_execution_nodes()->comm_size != x_d->align_template->onto_nodes->comm_size) same_nodes = 0;
   if(_XMP_get_execution_nodes()->comm_size != a_d->align_template->onto_nodes->comm_size) same_nodes = 0;
   if(_XMP_get_execution_nodes()->comm_size != b_d->align_template->onto_nodes->comm_size) same_nodes = 0;

   /* duplicate? */
   duplicate = 0;
   for(i=0; i<x_d->dim; i++){
      if(x_d->info[i].align_template_index >= 0){
         duplicate++;
      }
   }
   if(duplicate >= x_d->align_template->onto_nodes->dim) duplicate = 0;

   /* distribute & align check */
   same_align = 0;
   if(same_nodes){
      for(i=0; i<x_d->dim; i++){
         if(x_d->info[i].align_manner == _XMP_N_ALIGN_BLOCK ||
            x_d->info[i].align_manner == _XMP_N_ALIGN_GBLOCK){
            if(x_d->info[i].align_manner == a_d->info[i].align_manner &&
               x_d->info[i].align_manner == b_d->info[i].align_manner &&
               x_d->info[i].align_subscript == 0 &&
               a_d->info[i].align_subscript == 0 &&
               b_d->info[i].align_subscript == 0 &&
               x_d->info[i].ser_size == x_d->align_template->info[x_d->info[i].align_template_index].ser_size &&
               a_d->info[i].ser_size == a_d->align_template->info[a_d->info[i].align_template_index].ser_size &&
               b_d->info[i].ser_size == b_d->align_template->info[b_d->info[i].align_template_index].ser_size){
               same_align++;
               dist_dim = i;
            } else {
               same_align = 0;
               break;
            }
         }
      }
   }

   /* GBLOCK mapping check */
   if(same_align == 1 && x_d->info[dist_dim].align_manner == _XMP_N_ALIGN_GBLOCK){
      _XMP_template_chunk_t *x_chunk = &(x_d->align_template->chunk[x_d->info[dist_dim].align_template_index]);
      if(dist_dim == 0){
         _XMP_template_chunk_t *a_chunk = &(a_d->align_template->chunk[a_d->info[dist_dim].align_template_index]);
         for(i=0; i<x_d->align_template->onto_nodes->comm_size; i++){
            if(x_chunk->mapping_array[i] != a_chunk->mapping_array[i]){
               same_align = 0;
               break;
            }
         }
      } else {
         _XMP_template_chunk_t *b_chunk = &(b_d->align_template->chunk[b_d->info[dist_dim].align_template_index]);
         for(i=0; i<x_d->align_template->onto_nodes->comm_size; i++){
            if(x_chunk->mapping_array[i] != b_chunk->mapping_array[i]){
               same_align = 0;
               break;
            }
         }
      }
   } else if(same_align == 2){
      if(x_d->info[0].align_manner == _XMP_N_ALIGN_GBLOCK){
         _XMP_template_chunk_t *x_chunk = &(x_d->align_template->chunk[x_d->info[0].align_template_index]);
         _XMP_template_chunk_t *a_chunk = &(a_d->align_template->chunk[a_d->info[0].align_template_index]);
         for(i=0; i<x_d->align_template->onto_nodes->comm_size; i++){
            if(x_chunk->mapping_array[i] != a_chunk->mapping_array[i]){
               same_align = 0;
               break;
            }
         }
      }
      if(x_d->info[1].align_manner == _XMP_N_ALIGN_GBLOCK){
         _XMP_template_chunk_t *x_chunk = &(x_d->align_template->chunk[x_d->info[1].align_template_index]);
         _XMP_template_chunk_t *b_chunk = &(b_d->align_template->chunk[b_d->info[1].align_template_index]);
         for(i=0; i<x_d->align_template->onto_nodes->comm_size; i++){
            if(x_chunk->mapping_array[i] != b_chunk->mapping_array[i]){
               same_align = 0;
               break;
            }
         }
      }
   }
   
   /*  */
   if(same_nodes && !duplicate && same_align == 1){
      xmp_matmul_allgather(x_d, a_d, b_d, dist_dim);
   } else if(xmpf_running && same_nodes && !duplicate && same_align == 2){
      xmp_matmul_blockf(x_d, a_d, b_d);
   } else if(!xmpf_running && same_nodes && !duplicate && same_align == 2){
      xmp_matmul_blockc(x_d, a_d, b_d);
   } else {
      xmp_matmul_no_opt(x_d, a_d, b_d);
   }
}

void xmpf_matmul(void *x_p, void *a_p, void *b_p)
{
   xmpf_running = 1;
   xmp_matmul(x_p, a_p, b_p);
   xmpf_running = 0;
}
