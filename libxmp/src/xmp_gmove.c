/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include <stdarg.h>
#include <string.h>
#include "mpi.h"
#include "xmp_internal.h"
#include "xmp_math_function.h"

typedef struct _XMP_bcast_array_section_info_type {
  int lower;
  int upper;
  int stride;
} _XMP_bcast_array_section_info_t;

// FIXME do not use this function
static int _XMP_convert_rank_array_to_rank(_XMP_nodes_t *nodes, int *rank_array) {
  _Bool is_valid = false;
  int acc_rank = 0;
  int acc_nodes_size = 1;
  int nodes_dim = nodes->dim;
  for (int i = 0; i < nodes_dim; i++) {
    int rank = rank_array[i];

    if (rank != _XMP_N_INVALID_RANK) {
      is_valid = true;
      acc_rank += rank * acc_nodes_size;
      acc_nodes_size *= nodes->info[i].size;
    }
  }

  if (is_valid) {
    return acc_rank;
  }
  else {
    return _XMP_N_INVALID_RANK;
  }
}

static int _XMP_calc_gmove_array_owner_rank_SCALAR(_XMP_array_t *array, int *ref_index) {
  _XMP_template_t *template = array->align_template;
  _XMP_nodes_t *nodes = template->onto_nodes;

  int nodes_dim = nodes->dim;
  int rank_array[nodes_dim];
  for (int i = 0; i < nodes_dim; i++) {
    rank_array[i] = _XMP_N_INVALID_RANK;
  }

  int array_dim = array->dim;
  for (int i = 0; i < array_dim; i++) {
    _XMP_array_info_t *ai = &(array->info[i]);
    if (ai->align_manner != _XMP_N_ALIGN_NOT_ALIGNED) {
      int template_index = ai->align_template_index;

      _XMP_template_chunk_t *chunk = ai->align_template_chunk;
      if (chunk->dist_manner != _XMP_N_DIST_DUPLICATION) {
        int nodes_index = chunk->onto_nodes_index;
        rank_array[nodes_index] = _XMP_calc_template_owner_SCALAR(template, template_index,
                                                                  ref_index[i] + ai->align_subscript);
      }
    }
  }

  return _XMP_calc_linear_rank_on_exec_nodes(nodes, rank_array, _XMP_get_execution_nodes());
}

static void _XMP_gmove_bcast_SCALAR(_XMP_array_t *array, void *dst_addr, void *src_addr,
                                    size_t type_size, int src_rank) {
  _XMP_nodes_t *onto_nodes = (array->align_template)->onto_nodes;
  _XMP_nodes_t *exec_nodes = _XMP_get_execution_nodes();
  _XMP_ASSERT(exec_nodes->is_member);

  int my_rank = array->align_comm_rank;
  if ((exec_nodes == onto_nodes) ||
      ((exec_nodes == _XMP_world_nodes) && (exec_nodes->comm_size == onto_nodes->comm_size))) {
    _XMP_ASSERT(array->is_align_comm_member);

    if (src_rank == my_rank) {
      memcpy(dst_addr, src_addr, type_size);
    }

    MPI_Bcast(dst_addr, type_size, MPI_BYTE, src_rank, *((MPI_Comm *)array->align_comm));
  }
  else {
    MPI_Comm *exec_comm = exec_nodes->comm;

    int is_root = 0;
    if (src_rank == my_rank) {
      is_root = 1;
    }

    int num_roots = 0;
    MPI_Allreduce(&is_root, &num_roots, 1, MPI_INT, MPI_SUM, *exec_comm);
    if (num_roots == 0) {
      _XMP_fatal("no root for gmove broadcast");
    }

    MPI_Comm root_comm;
    MPI_Comm_split(*exec_comm, is_root, _XMP_world_rank, &root_comm);

    int color, key;
    if (is_root) {
      memcpy(dst_addr, src_addr, type_size);

      MPI_Comm_rank(root_comm, &color);
      key = 0;
    }
    else {
      color = _XMP_world_rank % num_roots;
      key = _XMP_world_rank + 1;
    }

    MPI_Comm gmove_comm;
    MPI_Comm_split(*exec_comm, color, key, &gmove_comm);

    MPI_Bcast(dst_addr, type_size, MPI_BYTE, 0, gmove_comm);

    MPI_Comm_free(&root_comm);
    MPI_Comm_free(&gmove_comm);
  }
}

static int _XMP_check_gmove_array_ref_inclusion_SCALAR(_XMP_array_t *array, int array_index, int ref_index) {
  _XMP_ASSERT(!(array->align_template)->is_owner);

  _XMP_array_info_t *ai = &(array->info[array_index]);
  if (ai->align_manner == _XMP_N_ALIGN_NOT_ALIGNED) {
    return _XMP_N_INT_TRUE;
  } else {
    int template_ref_index = ref_index + ai->align_subscript;
    return _XMP_check_template_ref_inclusion(template_ref_index, template_ref_index, 1,
                                             array->align_template, ai->align_template_index);
  }
}

/*
static int _XMP_calc_gmove_target_nodes_size(_XMP_nodes_t *nodes, int *rank_array) {
  int acc = 1;
  int nodes_dim = nodes->dim;
  for (int i = 0; i < nodes_dim; i++) {
    int rank = rank_array[i];

    if (rank == _XMP_N_INVALID_RANK) {
      acc *= nodes->info[i].size;
    }
  }

  return acc;
}
*/

static _Bool _XMP_calc_local_copy_template_BLOCK(_XMP_template_chunk_t *chunk,
                                                 long long *lower, long long *upper, int s) {
  long long l = *lower;
  long long u = *upper;
  long long template_lower = chunk->par_lower;
  long long template_upper = chunk->par_upper;

  if (s != 1) {
    int dst_mod = _XMP_modi_ll_i(l, s);
    // normalize template lower
    int lower_mod = _XMP_modi_ll_i(template_lower, s);
    if (lower_mod != dst_mod) {
      if (lower_mod < dst_mod) {
        template_lower += (dst_mod - lower_mod);
      }
      else {
        template_lower += (s - lower_mod + dst_mod);
      }
    }

    if (template_lower > template_upper) return false;
  }

  // calc lower
  if (l < template_lower) {
    *lower = template_lower;
  }
  else if (template_upper < l) {
    return false;
  }
  else  {
    *lower = l;
  }

  // calc upper
  if (u < template_lower) {
    return false;
  }
  else if (template_upper < u) {
    *upper = template_upper;
  }
  else {
    *upper = u;
  }

  return true;
}

// XXX used when ref_stride is 1
static _Bool _XMP_calc_local_copy_template_CYCLIC1(_XMP_template_chunk_t *chunk,
                                                   long long *lower, long long u, int *stride) {
  long long l = *lower;
  long long template_lower = chunk->par_lower;
  int nodes_size = chunk->onto_nodes_info->size;

  // calc lower
  int dst_mod = _XMP_modi_ll_i(template_lower, nodes_size);
  int lower_mod = _XMP_modi_ll_i(l, nodes_size);
  if (lower_mod != dst_mod) {
    if (lower_mod < dst_mod) {
      l += (dst_mod - lower_mod);
    }
    else {
      l += (nodes_size - lower_mod + dst_mod);
    }
  }

  if (u < l) {
    return false;
  }
  else {
    *lower = l;
  }

  // calc stride;
  *stride = nodes_size;

  return true;
}

static _Bool _XMP_calc_local_copy_home_ref(_XMP_array_t *dst_array, int dst_dim_index,
                                           int *dst_l, int *dst_u, int *dst_s,
                                           int *src_l, int *src_u, int *src_s) {
  if (_XMP_M_COUNT_TRIPLETi(*dst_l, *dst_u, *dst_s) != _XMP_M_COUNT_TRIPLETi(*src_l, *src_u, *src_s)) {
    _XMP_fatal("wrong assign statement"); // FIXME fix error msg
  }

  _XMP_array_info_t *dst_array_info = &(dst_array->info[dst_dim_index]);
  if ((dst_array_info->align_template_index) == _XMP_N_NO_ALIGNED_TEMPLATE) {
    return true;
  }
  else {
    long long align_subscript = dst_array_info->align_subscript;
    long long l = *dst_l + align_subscript;
    long long u = *dst_u + align_subscript;
    int s = *dst_s;

    _XMP_template_chunk_t *chunk = dst_array_info->align_template_chunk;
    switch (chunk->dist_manner) {
      case _XMP_N_DIST_DUPLICATION:
        return true;
      case _XMP_N_DIST_BLOCK:
        {
          _Bool res = _XMP_calc_local_copy_template_BLOCK(chunk, &l, &u, s);
          if (res) {
            int new_dst_l = l - align_subscript;
            int new_dst_u = u - align_subscript;

            // update src ref
            *src_l += (((new_dst_l - (*dst_l)) / (*dst_s)) * (*src_s));
            *src_u = (*src_l) + ((_XMP_M_COUNT_TRIPLETi(new_dst_l, new_dst_u, s) - 1) * (*src_s));

            // update dst ref
            *dst_l = new_dst_l;
            *dst_u = new_dst_u;
          }

          return res;
        }
      case _XMP_N_DIST_CYCLIC:
        {
          if (s == 1) {
            _Bool res = _XMP_calc_local_copy_template_CYCLIC1(chunk, &l, u, &s);
            if (res) {
              int new_dst_l = l - align_subscript;
              int new_dst_s = s;

              // update src ref
              *src_l += ((new_dst_l - (*dst_l)) * (*src_s));
              *src_s *= new_dst_s;

              // update dst ref
              *dst_l = new_dst_l;
              *dst_s = new_dst_s;
            }

            return res;
          }
          else {
            // FIXME
            _XMP_fatal("not implemented yet");
            return false; // XXX dummy;
          }
        }
      default:
        _XMP_fatal("unknown distribute manner");
        return false; // XXX dummy;
    }
  }
}

static void _XMP_calc_array_local_index_triplet(_XMP_array_t *array,
                                                int dim_index, int *lower, int *upper, int *stride) {
  _XMP_array_info_t *array_info = &(array->info[dim_index]);
  if ((array_info->align_template_index) != _XMP_N_NO_ALIGNED_TEMPLATE) {
    int dist_manner = (array_info->align_template_chunk)->dist_manner;
    switch (array_info->shadow_type) {
      case _XMP_N_SHADOW_NONE:
        {
          switch (dist_manner) {
            case _XMP_N_DIST_BLOCK:
              {
                *lower -= (*(array_info->temp0));
                *upper -= (*(array_info->temp0));
                *stride = 1;
              } break;
            case _XMP_N_DIST_CYCLIC:
              {
                *lower /= (*(array_info->temp0));
                *upper /= (*(array_info->temp0));
                *stride = 1;
              } break;
            default:
              _XMP_fatal("wrong distribute manner for normal shadow");
          }
        } break;
      case _XMP_N_SHADOW_NORMAL:
        {
          switch (dist_manner) {
            case _XMP_N_DIST_BLOCK:
              {
                *lower -= (*(array_info->temp0));
                *upper -= (*(array_info->temp0));
                *stride = 1;
              } break;
            // FIXME normal shadow is not allowed in cyclic distribution
            default:
              _XMP_fatal("wrong distribute manner for normal shadow");
          }
        } break;
      case _XMP_N_SHADOW_FULL:
        return;
      default:
        _XMP_fatal("unknown shadow type");
    }
  }
}

// ----- gmove scalar to scalar --------------------------------------------------------------------------------------------------
void _XMP_gmove_BCAST_SCALAR(void *dst_addr, void *src_addr, _XMP_array_t *array, ...) {
  va_list args;
  va_start(args, array);
  int src_rank;
  {
    int array_dim = array->dim;
    int ref_index[array_dim];
    for (int i = 0; i < array_dim; i++) {
      ref_index[i] = va_arg(args, int);
    }
    src_rank = _XMP_calc_gmove_array_owner_rank_SCALAR(array, ref_index);
  }
  va_end(args);

  size_t type_size = array->type_size;

  if (src_rank == _XMP_N_INVALID_RANK) {
    // local copy
    memcpy(dst_addr, src_addr, type_size);
  }
  else {
    // broadcast
    _XMP_gmove_bcast_SCALAR(array, dst_addr, src_addr, type_size, src_rank);
  }
}

int _XMP_gmove_HOMECOPY_SCALAR(_XMP_array_t *array, ...) {
  if (!array->is_allocated) {
    return _XMP_N_INT_FALSE;
  }

  _XMP_ASSERT((array->align_template)->is_distributed);
  _XMP_ASSERT((array->align_template)->is_owner);

  va_list args;
  va_start(args, array);
  int execHere = _XMP_N_INT_TRUE;
  int ref_dim = array->dim;
  for (int i = 0; i < ref_dim; i++) {
    int ref_index = va_arg(args, int);

    execHere = execHere && _XMP_check_gmove_array_ref_inclusion_SCALAR(array, i, ref_index);
  }
  va_end(args);

  return execHere;
}

void _XMP_gmove_SENDRECV_SCALAR(void *dst_addr, void *src_addr,
                                _XMP_array_t *dst_array, _XMP_array_t *src_array, ...) {
  va_list args;
  va_start(args, src_array);
  int dst_rank;
  {
    int dst_array_dim = dst_array->dim;
    int dst_ref_index[dst_array_dim];
    for (int i = 0; i < dst_array_dim; i++) {
      dst_ref_index[i] = va_arg(args, int);
    }
    dst_rank = _XMP_calc_gmove_array_owner_rank_SCALAR(dst_array, dst_ref_index);
  }

  int src_rank;
  {
    int src_array_dim = src_array->dim;
    int src_ref_index[src_array_dim];
    for (int i = 0; i < src_array_dim; i++) {
      src_ref_index[i] = va_arg(args, int);
    }
    src_rank = _XMP_calc_gmove_array_owner_rank_SCALAR(src_array, src_ref_index);
  }
  va_end(args);

  _XMP_ASSERT(dst_array->type_size == src_array->type_size); // FIXME checked by compiler
  size_t type_size = dst_array->type_size;

  if (dst_rank == _XMP_N_INVALID_RANK) {
    if (src_rank == _XMP_N_INVALID_RANK) {
      // local copy
      memcpy(dst_addr, src_addr, type_size);
    }
    else {
      // broadcast
      _XMP_gmove_bcast_SCALAR(src_array, dst_addr, src_addr, type_size, src_rank);
    }
  }
  else {
    if (src_rank == _XMP_N_INVALID_RANK) {
      // local copy on dst_rank
      if (dst_rank == dst_array->align_comm_rank) {
        memcpy(dst_addr, src_addr, type_size);
      }
    }
    else {
      _XMP_nodes_t *exec_nodes = _XMP_get_execution_nodes();
      _XMP_ASSERT(exec_nodes->is_member);

      MPI_Comm *exec_comm = exec_nodes->comm;

      int is_src = 0, num_srcs = 0;
      MPI_Comm src_comm;
      {
        if (src_rank == src_array->align_comm_rank) {
          is_src = 1;
        }

        MPI_Allreduce(&is_src, &num_srcs, 1, MPI_INT, MPI_SUM, *exec_comm);
        if (num_srcs == 0) {
          _XMP_fatal("no source for gmove send/recv");
        }

        MPI_Comm_split(*exec_comm, is_src, _XMP_world_rank, &src_comm);
      }

      int is_dst = 0;
      MPI_Comm gmove_comm;
      {
        int color = num_srcs, key = 0;
        if (dst_rank == dst_array->align_comm_rank) {
          is_dst = 1;

          color = _XMP_world_rank % num_srcs;
          key = _XMP_world_rank + 1;
        }

        // overwrite color, key
        if (is_src) {
          MPI_Comm_rank(src_comm, &color);
          key = 0;
        }

        MPI_Comm_split(*exec_comm, color, key, &gmove_comm);

        if (color == num_srcs) {
          goto EXIT_AFTER_CLEAN_UP;
        }
      }

      int gmove_comm_size;
      MPI_Comm_size(gmove_comm, &gmove_comm_size);

      if (gmove_comm_size == 1) {
        if (is_dst) {
          memcpy(dst_addr, src_addr, type_size);
        }
      }
      else if (gmove_comm_size == 2) {
        if (is_src) {
          MPI_Send(src_addr, type_size, MPI_BYTE, 1, _XMP_N_MPI_TAG_GMOVE, gmove_comm);
        }

        if (is_dst) {
          if (is_src) {
            memcpy(dst_addr, src_addr, type_size);
          }
          else {
            MPI_Status stat;
            MPI_Recv(dst_addr, type_size, MPI_BYTE, 0, _XMP_N_MPI_TAG_GMOVE, gmove_comm, &stat);
          }
        }
      }
      else {
        void *temp_buffer = _XMP_alloc(type_size);

        if (is_src) {
          memcpy(temp_buffer, src_addr, type_size);
        }

        MPI_Bcast(temp_buffer, type_size, MPI_BYTE, 0, gmove_comm);

        if (is_dst) {
          memcpy(dst_addr, temp_buffer, type_size);
        }

        _XMP_free(temp_buffer);
      }

EXIT_AFTER_CLEAN_UP:
      MPI_Comm_free(&src_comm);
      MPI_Comm_free(&gmove_comm);
    }
  }
}

// ----- gmove vector to vector --------------------------------------------------------------------------------------------------
void _XMP_gmove_LOCALCOPY_ARRAY(int type, size_t type_size, ...) {
  va_list args;
  va_start(args, type_size);

  // get dst info
  void *dst_addr = va_arg(args, void *);
  int dst_dim = va_arg(args, int);
  int dst_l[dst_dim], dst_u[dst_dim], dst_s[dst_dim]; unsigned long long dst_d[dst_dim];
  unsigned long long dst_buffer_elmts = 1;
  for (int i = 0; i < dst_dim; i++) {
    dst_l[i] = va_arg(args, int);
    dst_u[i] = va_arg(args, int);
    dst_s[i] = va_arg(args, int);
    dst_d[i] = va_arg(args, unsigned long long);
    _XMP_normalize_array_section(&(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
    dst_buffer_elmts *= _XMP_M_COUNT_TRIPLETi(dst_l[i], dst_u[i], dst_s[i]);
  }

  // get src info
  void *src_addr = va_arg(args, void *);
  int src_dim = va_arg(args, int);
  int src_l[src_dim], src_u[src_dim], src_s[src_dim]; unsigned long long src_d[src_dim];
  unsigned long long src_buffer_elmts = 1;
  for (int i = 0; i < src_dim; i++) {
    src_l[i] = va_arg(args, int);
    src_u[i] = va_arg(args, int);
    src_s[i] = va_arg(args, int);
    src_d[i] = va_arg(args, unsigned long long);
    _XMP_normalize_array_section(&(src_l[i]), &(src_u[i]), &(src_s[i]));
    src_buffer_elmts *= _XMP_M_COUNT_TRIPLETi(src_l[i], src_u[i], src_s[i]);
  }

  va_end(args);
  
  // alloc buffer
  if (dst_buffer_elmts != src_buffer_elmts) {
    _XMP_fatal("wrong assign statement"); // FIXME fix error msg
  }

  void *buffer = _XMP_alloc(dst_buffer_elmts * type_size);

  // pack/unpack
  if (type == _XMP_N_TYPE_NONBASIC) {
    _XMP_pack_array_GENERAL(buffer, src_addr, type_size, src_dim, src_l, src_u, src_s, src_d);
    _XMP_unpack_array_GENERAL(dst_addr, buffer, type_size, dst_dim, dst_l, dst_u, dst_s, dst_d);
  }
  else {
    _XMP_pack_array_BASIC(buffer, src_addr, type, src_dim, src_l, src_u, src_s, src_d);
    _XMP_unpack_array_BASIC(dst_addr, buffer, type, dst_dim, dst_l, dst_u, dst_s, dst_d);
  }

  // free buffer
  _XMP_free(buffer);
}

void _XMP_gmove_BCAST_ARRAY(_XMP_array_t *src_array, int type, size_t type_size, ...) {
  _XMP_ASSERT((src_array->align_template)->is_owner);

  va_list args;
  va_start(args, type_size);

  // get dst info
  void *dst_addr = va_arg(args, void *);
  int dst_dim = va_arg(args, int);
  int dst_l[dst_dim], dst_u[dst_dim], dst_s[dst_dim]; unsigned long long dst_d[dst_dim];
  for (int i = 0; i < dst_dim; i++) {
    dst_l[i] = va_arg(args, int);
    dst_u[i] = va_arg(args, int);
    dst_s[i] = va_arg(args, int);
    dst_d[i] = va_arg(args, unsigned long long);
    _XMP_normalize_array_section(&(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
  }

  // get src info
  void *src_addr = va_arg(args, void *);
  int src_dim = va_arg(args, int);
  int src_l[src_dim], src_u[src_dim], src_s[src_dim]; unsigned long long src_d[src_dim];
  for (int i = 0; i < src_dim; i++) {
    src_l[i] = va_arg(args, int);
    src_u[i] = va_arg(args, int);
    src_s[i] = va_arg(args, int);
    src_d[i] = va_arg(args, unsigned long long);
    _XMP_normalize_array_section(&(src_l[i]), &(src_u[i]), &(src_s[i]));
  }

  va_end(args);

  // calc index ref
  int is_root = _XMP_N_INT_TRUE;
  int dst_dim_index = 0;
  unsigned long long dst_buffer_elmts = 1;
  unsigned long long src_buffer_elmts = 1;
  for (int i = 0; i < src_dim; i++) {
    int src_elmts = _XMP_M_COUNT_TRIPLETi(src_l[i], src_u[i], src_s[i]);
    if (src_elmts == 1) {
      if(!_XMP_check_gmove_array_ref_inclusion_SCALAR(src_array, i, src_l[i])) {
        is_root = _XMP_N_INT_FALSE;
        break;
      }
    }
    else {
      src_buffer_elmts *= src_elmts;

      int dst_elmts;
      do {
        dst_elmts = _XMP_M_COUNT_TRIPLETi(dst_l[dst_dim_index], dst_u[dst_dim_index], dst_s[dst_dim_index]);
        dst_dim_index++;
      } while (dst_elmts == 1);

      int j = dst_dim_index - 1;
      if (_XMP_calc_local_copy_home_ref(src_array, i, &(src_l[i]), &(src_u[i]), &(src_s[i]),
                                                             &(dst_l[j]), &(dst_u[j]), &(dst_s[j]))) {
        dst_buffer_elmts *= dst_elmts;
      }
      else {
        is_root = _XMP_N_INT_FALSE;
        break;
      }
    }

    _XMP_calc_array_local_index_triplet(src_array, i, &(src_l[i]), &(src_u[i]), &(src_s[i]));
  }

  // bcast data
  void *pack_buffer = NULL;
  if (is_root) {
    for (int i = dst_dim_index; i < dst_dim; i++) {
      dst_buffer_elmts *= _XMP_M_COUNT_TRIPLETi(dst_l[i], dst_u[i], dst_s[i]);
    }

    if (dst_buffer_elmts != src_buffer_elmts) {
      _XMP_fatal("wrong assign statement"); // FIXME fix error msg
    }

    pack_buffer = _XMP_alloc(src_buffer_elmts * type_size);
    if (type == _XMP_N_TYPE_NONBASIC) {
      _XMP_pack_array_GENERAL(pack_buffer, src_addr, type_size, src_dim, src_l, src_u, src_s, src_d);
    }
    else {
      _XMP_pack_array_BASIC(pack_buffer, src_addr, type, src_dim, src_l, src_u, src_s, src_d);
    }
  }

  _XMP_nodes_t *exec_nodes = _XMP_get_execution_nodes();
  MPI_Comm *exec_nodes_comm = exec_nodes->comm;
  int exec_nodes_size = exec_nodes->comm_size;
  int exec_nodes_rank = exec_nodes->comm_rank;

  int root_nodes[exec_nodes_size];
  MPI_Allgather(&is_root, 1, MPI_INT, root_nodes, 1, MPI_INT, *exec_nodes_comm);

  _XMP_bcast_array_section_info_t bcast_info[dst_dim];

  MPI_Datatype mpi_datatype;
  MPI_Type_contiguous(type_size, MPI_BYTE, &mpi_datatype);
  MPI_Type_commit(&mpi_datatype);

  int bcast_l[dst_dim], bcast_u[dst_dim], bcast_s[dst_dim];
  unsigned long long bcast_elmts;
  for (int i = 0; i < exec_nodes_size; i++) {
    if (root_nodes[i]) {
      if (i == exec_nodes_rank) {
        for (int j = 0; j < dst_dim; j++) {
          bcast_info[j].lower = dst_l[j];
          bcast_info[j].upper = dst_u[j];
          bcast_info[j].stride = dst_s[j];
        }
      }

      MPI_Bcast(bcast_info, sizeof(_XMP_bcast_array_section_info_t) * dst_dim, MPI_BYTE, i, *exec_nodes_comm);

      bcast_elmts = 1;
      for (int j = 0; j < dst_dim; j++) {
        bcast_l[j] = bcast_info[j].lower;
        bcast_u[j] = bcast_info[j].upper;
        bcast_s[j] = bcast_info[j].stride;
        bcast_elmts *= _XMP_M_COUNT_TRIPLETi(bcast_l[j], bcast_u[j], bcast_s[j]);
      }

      void *bcast_buffer;
      if (i == exec_nodes_rank) {
        bcast_buffer = pack_buffer;
      }
      else {
        bcast_buffer = _XMP_alloc(bcast_elmts * type_size);
      }
      MPI_Bcast(bcast_buffer, bcast_elmts, mpi_datatype, i, *exec_nodes_comm);

      if (type == _XMP_N_TYPE_NONBASIC) {
        _XMP_unpack_array_GENERAL(dst_addr, bcast_buffer, type_size, dst_dim, bcast_l, bcast_u, bcast_s, dst_d);
      }
      else {
        _XMP_unpack_array_BASIC(dst_addr, bcast_buffer, type, dst_dim, bcast_l, bcast_u, bcast_s, dst_d);
      }
      _XMP_free(bcast_buffer);
    }
  }

  MPI_Type_free(&mpi_datatype);
}

void _XMP_gmove_HOMECOPY_ARRAY(_XMP_array_t *dst_array, int type, size_t type_size, ...) {
  if (!dst_array->is_allocated) {
    return;
  }

  _XMP_ASSERT((dst_array->align_template)->is_owner);

  va_list args;
  va_start(args, type_size);

  // get dst info
  void *dst_addr = va_arg(args, void *);
  int dst_dim = va_arg(args, int);
  int dst_l[dst_dim], dst_u[dst_dim], dst_s[dst_dim]; unsigned long long dst_d[dst_dim];
  for (int i = 0; i < dst_dim; i++) {
    dst_l[i] = va_arg(args, int);
    dst_u[i] = va_arg(args, int);
    dst_s[i] = va_arg(args, int);
    dst_d[i] = va_arg(args, unsigned long long);
    _XMP_normalize_array_section(&(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
  }

  // get src info
  void *src_addr = va_arg(args, void *);
  int src_dim = va_arg(args, int);
  int src_l[src_dim], src_u[src_dim], src_s[src_dim]; unsigned long long src_d[src_dim];
  for (int i = 0; i < src_dim; i++) {
    src_l[i] = va_arg(args, int);
    src_u[i] = va_arg(args, int);
    src_s[i] = va_arg(args, int);
    src_d[i] = va_arg(args, unsigned long long);
    _XMP_normalize_array_section(&(src_l[i]), &(src_u[i]), &(src_s[i]));
  }

  va_end(args);

  // calc index ref
  int src_dim_index = 0;
  unsigned long long dst_buffer_elmts = 1;
  unsigned long long src_buffer_elmts = 1;
  for (int i = 0; i < dst_dim; i++) {
    int dst_elmts = _XMP_M_COUNT_TRIPLETi(dst_l[i], dst_u[i], dst_s[i]);
    if (dst_elmts == 1) {
      if(!_XMP_check_gmove_array_ref_inclusion_SCALAR(dst_array, i, dst_l[i])) {
        return;
      }
    }
    else {
      dst_buffer_elmts *= dst_elmts;

      int src_elmts;
      do {
        src_elmts = _XMP_M_COUNT_TRIPLETi(src_l[src_dim_index], src_u[src_dim_index], src_s[src_dim_index]);
        src_dim_index++;
      } while (src_elmts == 1);

      int j = src_dim_index - 1;
      if (_XMP_calc_local_copy_home_ref(dst_array, i, &(dst_l[i]), &(dst_u[i]), &(dst_s[i]),
                                                             &(src_l[j]), &(src_u[j]), &(src_s[j]))) {
        src_buffer_elmts *= src_elmts;
      }
      else {
        return;
      }
    }

    _XMP_calc_array_local_index_triplet(dst_array, i, &(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
  }

  for (int i = src_dim_index; i < src_dim; i++) {
    src_buffer_elmts *= _XMP_M_COUNT_TRIPLETi(src_l[i], src_u[i], src_s[i]);
  }

  // alloc buffer
  if (dst_buffer_elmts != src_buffer_elmts) {
    _XMP_fatal("wrong assign statement"); // FIXME fix error msg
  }

  void *buffer = _XMP_alloc(dst_buffer_elmts * type_size);

  // pack/unpack
  if (type == _XMP_N_TYPE_NONBASIC) {
    _XMP_pack_array_GENERAL(buffer, src_addr, type_size, src_dim, src_l, src_u, src_s, src_d);
    _XMP_unpack_array_GENERAL(dst_addr, buffer, type_size, dst_dim, dst_l, dst_u, dst_s, dst_d);
  }
  else {
    _XMP_pack_array_BASIC(buffer, src_addr, type, src_dim, src_l, src_u, src_s, src_d);
    _XMP_unpack_array_BASIC(dst_addr, buffer, type, dst_dim, dst_l, dst_u, dst_s, dst_d);
  }

  // free buffer
  _XMP_free(buffer);
}

// FIXME temporary implementation
static int _XMP_calc_SENDRECV_owner(_XMP_array_t *array, int *lower, int *upper, int *stride) {
  _XMP_template_t *template = array->align_template;
  _XMP_nodes_t *nodes = template->onto_nodes;

  int nodes_dim = nodes->dim;
  int rank_array[nodes_dim];
  for (int i = 0; i < nodes_dim; i++) {
    rank_array[i] = _XMP_N_INVALID_RANK;
  }

  int array_dim = array->dim;
  for (int i = 0; i < array_dim; i++) {
    _XMP_array_info_t *ai = &(array->info[i]);
    int template_index = ai->align_template_index;
    if (template_index != _XMP_N_NO_ALIGNED_TEMPLATE) {
      if (_XMP_M_COUNT_TRIPLETi(lower[i], upper[i], stride[i]) == 1) {
        int nodes_index = (ai->align_template_chunk)->onto_nodes_index;
        if (nodes_index != _XMP_N_NO_ONTO_NODES) {
          int owner = _XMP_calc_template_owner_SCALAR(template, template_index, lower[i] + ai->align_subscript);
          if (owner != _XMP_N_INVALID_RANK) {
            rank_array[nodes_index] = owner;
          }
        }
      }
      else {
        if (((ai->align_template_chunk)->dist_manner) != _XMP_N_DIST_DUPLICATION) {
          return _XMP_N_INVALID_RANK;
        }
      }
    }
  }

  return _XMP_convert_rank_array_to_rank(nodes, rank_array);
}

// FIXME does not has complete function for general usage
static void _XMP_calc_SENDRECV_index_ref(int n, int target_rank, _XMP_array_t *array, int dim_index,
                                         int *lower, int *upper, int *stride) {
  _XMP_array_info_t *array_info = &(array->info[dim_index]);
  if ((array_info->align_template_index) == _XMP_N_NO_ALIGNED_TEMPLATE) {
    *lower = target_rank * n;
    *upper = ((target_rank + 1) * n) - 1;
    *stride = 1;
  }
  else {
    *lower = 0;
    *upper = n - 1;
    *stride = 1;
  }
}

// FIXME does not has complete function for general usage
static void _XMP_gmove_SENDRECV_all2all_2(void *dst_addr, void *src_addr,
                                                 _XMP_array_t *dst_array, _XMP_array_t *src_array,
                                                 int type, size_t type_size,
                                                 int *dst_lower, int *dst_upper, int *dst_stride, unsigned long long *dst_dim_acc,
                                                 int *src_lower, int *src_upper, int *src_stride, unsigned long long *src_dim_acc) {
  int dim = dst_array->dim;
  if (dim != src_array->dim) {
    _XMP_fatal("dst/src array should have the same dimension");
  }

  MPI_Status stat;
  MPI_Comm *comm = ((dst_array->align_template)->onto_nodes)->comm;
  int size = ((dst_array->align_template)->onto_nodes)->comm_size;
  int rank = ((dst_array->align_template)->onto_nodes)->comm_rank;

  MPI_Datatype mpi_datatype;
  MPI_Type_contiguous(type_size, MPI_BYTE, &mpi_datatype);
  MPI_Type_commit(&mpi_datatype);

  unsigned long long buffer_elmts = 1;
  int elmts_base = _XMP_M_COUNT_TRIPLETi(dst_lower[0], dst_upper[0], dst_stride[0]);
  int n = elmts_base/size;
  for (int i = 0; i < dim; i++) {
    int dst_elmts = _XMP_M_COUNT_TRIPLETi(dst_lower[i], dst_upper[i], dst_stride[i]);
    if (dst_elmts != elmts_base) {
      _XMP_fatal("limitation:every dimension should has the same size");
    }

    int src_elmts = _XMP_M_COUNT_TRIPLETi(src_lower[i], src_upper[i], src_stride[i]);
    if (src_elmts != elmts_base) {
      _XMP_fatal("limitation:every dimension should has the same size");
    }

    buffer_elmts *= n;
  }

  int dst_l[dim], dst_u[dim], dst_s[dim];
  int src_l[dim], src_u[dim], src_s[dim];
  void *pack_buffer = _XMP_alloc(buffer_elmts * type_size);
  for(int src_rank = 0; src_rank < size; src_rank++) {
    if(src_rank == rank) {
      // send my data to each node
      for(int dst_rank = 0; dst_rank < size; dst_rank++) {
        if(dst_rank == rank) {
          for (int i = 0; i < dim; i++) {
            _XMP_calc_SENDRECV_index_ref(n, dst_rank, dst_array, i, &(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
            _XMP_calc_SENDRECV_index_ref(n, dst_rank, src_array, i, &(src_l[i]), &(src_u[i]), &(src_s[i]));
          }

          if (type == _XMP_N_TYPE_NONBASIC) {
            _XMP_pack_array_GENERAL(pack_buffer, src_addr, type_size, dim, src_l, src_u, src_s, src_dim_acc);
            _XMP_unpack_array_GENERAL(dst_addr, pack_buffer, type_size, dim, dst_l, dst_u, dst_s, dst_dim_acc);
          }
          else {
            _XMP_pack_array_BASIC(pack_buffer, src_addr, type, dim, src_l, src_u, src_s, src_dim_acc);
            _XMP_unpack_array_BASIC(dst_addr, pack_buffer, type, dim, dst_l, dst_u, dst_s, dst_dim_acc);
          }
        }
        else {
          for (int i = 0; i < dim; i++) {
            _XMP_calc_SENDRECV_index_ref(n, dst_rank, src_array, i, &(src_l[i]), &(src_u[i]), &(src_s[i]));
          }

          if (type == _XMP_N_TYPE_NONBASIC) {
            _XMP_pack_array_GENERAL(pack_buffer, src_addr, type_size, dim, src_l, src_u, src_s, src_dim_acc);
          }
          else {
            _XMP_pack_array_BASIC(pack_buffer, src_addr, type, dim, src_l, src_u, src_s, src_dim_acc);
          }

          MPI_Send(pack_buffer, buffer_elmts, mpi_datatype, dst_rank, _XMP_N_MPI_TAG_GMOVE, *comm);
        }
      }
    }
    else {
      MPI_Recv(pack_buffer, buffer_elmts, mpi_datatype, src_rank, _XMP_N_MPI_TAG_GMOVE, *comm, &stat);

      for (int i = 0; i < dim; i++) {
        _XMP_calc_SENDRECV_index_ref(n, src_rank, dst_array, i, &(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
      }

      if (type == _XMP_N_TYPE_NONBASIC) {
        _XMP_unpack_array_GENERAL(dst_addr, pack_buffer, type_size, dim, dst_l, dst_u, dst_s, dst_dim_acc);
      }
      else {
        _XMP_unpack_array_BASIC(dst_addr, pack_buffer, type, dim, dst_l, dst_u, dst_s, dst_dim_acc);
      }
    }
  }

  MPI_Type_free(&mpi_datatype);
  _XMP_free(pack_buffer);
}

// FIXME does not has complete function for general usage
void _XMP_gmove_SENDRECV_ARRAY(_XMP_array_t *dst_array, _XMP_array_t *src_array,
                               int type, size_t type_size, ...) {
  va_list args;
  va_start(args, type_size);

  // get dst info
  void *dst_addr = va_arg(args, void *);
  int dst_dim = va_arg(args, int);
  int dst_l[dst_dim], dst_u[dst_dim], dst_s[dst_dim]; unsigned long long dst_d[dst_dim];
  for (int i = 0; i < dst_dim; i++) {
    dst_l[i] = va_arg(args, int);
    dst_u[i] = va_arg(args, int);
    dst_s[i] = va_arg(args, int);
    dst_d[i] = va_arg(args, unsigned long long);
    _XMP_normalize_array_section(&(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
  }

  // get src info
  void *src_addr = va_arg(args, void *);
  int src_dim = va_arg(args, int);
  int src_l[src_dim], src_u[src_dim], src_s[src_dim]; unsigned long long src_d[src_dim];
  for (int i = 0; i < src_dim; i++) {
    src_l[i] = va_arg(args, int);
    src_u[i] = va_arg(args, int);
    src_s[i] = va_arg(args, int);
    src_d[i] = va_arg(args, unsigned long long);
    _XMP_normalize_array_section(&(src_l[i]), &(src_u[i]), &(src_s[i]));
  }

  va_end(args);

  _XMP_nodes_t *dst_nodes = (dst_array->align_template)->onto_nodes;
  _XMP_nodes_t *src_nodes = (src_array->align_template)->onto_nodes;

  int dst_rank = _XMP_calc_SENDRECV_owner(dst_array, dst_l, dst_u, dst_s);
  int src_rank = _XMP_calc_SENDRECV_owner(src_array, src_l, src_u, src_s);
  if ((dst_rank != _XMP_N_INVALID_RANK) && (src_rank != _XMP_N_INVALID_RANK)) {
    // send/recv FIXME limitation: arrays should be distributed by the same nodes
    if (dst_nodes != src_nodes) {
      _XMP_fatal("arrays used in a gmove directive should be distributed by the same nodes set");
    }

    // FIXME use execution nodes set
    _XMP_nodes_t *comm_nodes = dst_nodes;

    void *recv_buffer = NULL;
    void *send_buffer = NULL;

    MPI_Datatype mpi_datatype;
    MPI_Type_contiguous(type_size, MPI_BYTE, &mpi_datatype);
    MPI_Type_commit(&mpi_datatype);

    // irecv
    MPI_Request recv_req;
    if (dst_rank == dst_array->align_comm_rank) {
      unsigned long long recv_elmts = 1;
      for (int i = 0; i < dst_dim; i++) {
        recv_elmts *= _XMP_M_COUNT_TRIPLETi(dst_l[i], dst_u[i], dst_s[i]);
      }
      recv_buffer = _XMP_alloc(recv_elmts * type_size);

      MPI_Irecv(recv_buffer, recv_elmts, mpi_datatype, MPI_ANY_SOURCE, _XMP_N_MPI_TAG_GMOVE,
                *((MPI_Comm *)comm_nodes->comm), &recv_req);
    }

    // pack & send
    if (src_rank == src_array->align_comm_rank) {
      unsigned long long send_elmts = 1;
      for (int i = 0; i < src_dim; i++) {
        _XMP_calc_array_local_index_triplet(src_array, i, &(src_l[i]), &(src_u[i]), &(src_s[i]));
        send_elmts *= _XMP_M_COUNT_TRIPLETi(src_l[i], src_u[i], src_s[i]);
      }
      send_buffer = _XMP_alloc(send_elmts * type_size);
      if (type == _XMP_N_TYPE_NONBASIC) {
        _XMP_pack_array_GENERAL(send_buffer, src_addr, type_size, src_dim, src_l, src_u, src_s, src_d);
      }
      else {
        _XMP_pack_array_BASIC(send_buffer, src_addr, type, src_dim, src_l, src_u, src_s, src_d);
      }

      MPI_Send(send_buffer, send_elmts, mpi_datatype, dst_rank, _XMP_N_MPI_TAG_GMOVE, *((MPI_Comm *)comm_nodes->comm));
      _XMP_free(send_buffer);
    }

    // wait & unpack
    if (dst_rank == dst_array->align_comm_rank) {
      for (int i = 0; i < dst_dim; i++) {
        _XMP_calc_array_local_index_triplet(dst_array, i, &(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
      }

      MPI_Status recv_stat;
      MPI_Wait(&recv_req, &recv_stat);

      if (type == _XMP_N_TYPE_NONBASIC) {
        _XMP_unpack_array_GENERAL(dst_addr, recv_buffer, type_size, dst_dim, dst_l, dst_u, dst_s, dst_d);
      }
      else {
        _XMP_unpack_array_BASIC(dst_addr, recv_buffer, type, dst_dim, dst_l, dst_u, dst_s, dst_d);
      }
      _XMP_free(recv_buffer);
    }

    MPI_Type_free(&mpi_datatype);
  }
  else {
    if (dst_array == src_array) {
      unsigned long long dst_buffer_elmts = 1;
      for (int i = 0; i < dst_dim; i++) {
        _XMP_calc_array_local_index_triplet(dst_array, i, &(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
        dst_buffer_elmts *= _XMP_M_COUNT_TRIPLETi(dst_l[i], dst_u[i], dst_s[i]);
      }

      unsigned long long src_buffer_elmts = 1;
      for (int i = 0; i < src_dim; i++) {
        _XMP_calc_array_local_index_triplet(src_array, i, &(src_l[i]), &(src_u[i]), &(src_s[i]));
        src_buffer_elmts *= _XMP_M_COUNT_TRIPLETi(src_l[i], src_u[i], src_s[i]);
      }

      // alloc buffer
      if (dst_buffer_elmts != src_buffer_elmts) {
        _XMP_fatal("wrong assign statement"); // FIXME fix error msg
      }

      void *buffer = _XMP_alloc(dst_buffer_elmts * type_size);

      // pack/unpack
      if (type == _XMP_N_TYPE_NONBASIC) {
        _XMP_pack_array_GENERAL(buffer, src_addr, type_size, src_dim, src_l, src_u, src_s, src_d);
        _XMP_unpack_array_GENERAL(dst_addr, buffer, type_size, dst_dim, dst_l, dst_u, dst_s, dst_d);
      }
      else {
        _XMP_pack_array_BASIC(buffer, src_addr, type, src_dim, src_l, src_u, src_s, src_d);
        _XMP_unpack_array_BASIC(dst_addr, buffer, type, dst_dim, dst_l, dst_u, dst_s, dst_d);
      }

      // free buffer
      _XMP_free(buffer);
    }
    else {
      if (dst_dim == src_dim) {
        if (dst_dim == 1) {
          _XMP_array_info_t *ai = &(dst_array->info[0]);

          _XMP_push_comm(src_array->align_comm);
          _XMP_gmove_BCAST_ARRAY(src_array, type, type_size,
                                        dst_addr, dst_dim, ai->local_lower, ai->local_upper, ai->local_stride, dst_d[0],
                                        src_addr, src_dim, ai->par_lower, ai->par_upper, ai->par_stride, dst_d[0]);
          _XMP_pop_n_free_nodes_wo_finalize_comm();
        }
        else if (dst_dim == 2) {
          _XMP_gmove_SENDRECV_all2all_2(dst_addr, src_addr,
                                               dst_array, src_array,
                                               type, type_size,
                                               dst_l, dst_u, dst_s, dst_d,
                                               src_l, src_u, src_s, src_d);
        }
        else {
          _XMP_fatal("not implemented yet");
        }
      }
      else {
        _XMP_fatal("not implemented yet");
      }
    }
  }

  // FIXME delete this after bug fix
  _XMP_barrier_EXEC();
}
