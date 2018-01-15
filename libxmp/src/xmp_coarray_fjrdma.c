#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <assert.h>
#include <string.h>
#include "mpi.h"
#include "mpi-ext.h"
#include "xmp_internal.h"
#define _XMP_FJRDMA_MAX_SIZE  16777212
#define _XMP_FJRDMA_MAX_MEMID      511
#define _XMP_FJRDMA_MAX_MPUT      1993
#define _XMP_FJRDMA_MAX_MGET       100 /** This value is trial */
#define _XMP_FJRDMA_MAX_PUT         60 /** This value is trial */
#define _XMP_FJRDMA_MAX_GET         60 /** This value is trial */

static int _num_of_puts = 0, _num_of_gets = 0;
static int _memid = _XMP_FJRDMA_START_MEMID; // _memid = 0 is used for put/get operations.
                                             // _memid = 1 is used for post/wait operations.
                                             // _memid = 2 is used for sync images
static uint64_t _local_rdma_addr, *_remote_rdma_addr;
static unsigned int *_sync_images_table;

/** These variables are temporral **/
extern int _XMP_flag_put_nb;
extern int _XMP_flag_put_nb_rr;
extern int _XMP_flag_put_nb_rr_i;
#define _XMP_COARRAY_SEND_NIC_TMP_0 FJMPI_RDMA_LOCAL_NIC0
#define _XMP_COARRAY_SEND_NIC_TMP_1 FJMPI_RDMA_LOCAL_NIC1
#define _XMP_COARRAY_SEND_NIC_TMP_2 FJMPI_RDMA_LOCAL_NIC2
#define _XMP_COARRAY_SEND_NIC_TMP_3 FJMPI_RDMA_LOCAL_NIC3
#define _XMP_COARRAY_FLAG_NIC_TMP_0 (FJMPI_RDMA_LOCAL_NIC0 | FJMPI_RDMA_REMOTE_NIC0)
#define _XMP_COARRAY_FLAG_NIC_TMP_1 (FJMPI_RDMA_LOCAL_NIC1 | FJMPI_RDMA_REMOTE_NIC1)
#define _XMP_COARRAY_FLAG_NIC_TMP_2 (FJMPI_RDMA_LOCAL_NIC2 | FJMPI_RDMA_REMOTE_NIC2)
#define _XMP_COARRAY_FLAG_NIC_TMP_3 (FJMPI_RDMA_LOCAL_NIC3 | FJMPI_RDMA_REMOTE_NIC3)
#define _XMP_COARRAY_FLAG_NIC_TMP_i0 (FJMPI_RDMA_LOCAL_NIC0 | FJMPI_RDMA_REMOTE_NIC0 | FJMPI_RDMA_IMMEDIATE_RETURN)
#define _XMP_COARRAY_FLAG_NIC_TMP_i1 (FJMPI_RDMA_LOCAL_NIC1 | FJMPI_RDMA_REMOTE_NIC1 | FJMPI_RDMA_IMMEDIATE_RETURN)
#define _XMP_COARRAY_FLAG_NIC_TMP_i2 (FJMPI_RDMA_LOCAL_NIC2 | FJMPI_RDMA_REMOTE_NIC2 | FJMPI_RDMA_IMMEDIATE_RETURN)
#define _XMP_COARRAY_FLAG_NIC_TMP_i3 (FJMPI_RDMA_LOCAL_NIC3 | FJMPI_RDMA_REMOTE_NIC3 | FJMPI_RDMA_IMMEDIATE_RETURN)
/** End these variables are temporral **/ 

/******************************************************************/
/* DESCRIPTION : Set addresses                                    */
/* ARGUMENT    : [OUT] *addrs     : Addresses                     */
/*               [IN] *base_addr  : Base address                  */
/*               [IN] *array_info : Information of array          */
/*               [IN] dims        : Number of dimensions of array */
/*               [IN] chunk_size  : Chunk size for copy           */
/*               [IN] copy_elmts  : Num of elements for copy      */
/******************************************************************/
static void _XMP_set_coarray_addresses_with_chunk(uint64_t* addrs, const uint64_t base_addr, const _XMP_array_section_t* array_info, 
						  const int dims, const size_t chunk_size, const size_t copy_elmts)
{
  uint64_t stride_offset[dims], tmp[dims];

  // Temporally variables to reduce calculation for offset
  for(int i=0;i<dims;i++)
    stride_offset[i] = array_info[i].stride * array_info[i].distance;

  // array_info[dims-1].distance is an element size
  // chunk_size >= array_info[dims-1].distance
  switch (dims){
    int chunk_len;
  case 1:
    chunk_len = chunk_size / array_info[0].distance;
    for(size_t i=0,num=0;i<array_info[0].length;i+=chunk_len){
      addrs[num++] = stride_offset[0] * i + base_addr;
    }
    break;
  case 2:
    if(array_info[0].distance > chunk_size){ // array_info[0].distance > chunk_size >= array_info[1].distance
      chunk_len = chunk_size / array_info[1].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j+=chunk_len){
          tmp[1] = stride_offset[1] * j;
          addrs[num++] = tmp[0] + tmp[1] + base_addr;
        }
      }
    }
    else{                               // chunk_size >= array_info[0].distance
      chunk_len = chunk_size / array_info[0].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i+=chunk_len){
        addrs[num++] = stride_offset[0] * i + base_addr;
      }
    }
    break;
  case 3:
    if(array_info[1].distance > chunk_size){ // array_info[1].distance > chunk_size >= array_info[2].distance
      chunk_len = chunk_size / array_info[2].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(size_t k=0;k<array_info[2].length;k+=chunk_len){
            tmp[2] = stride_offset[2] * k;
            addrs[num++] = tmp[0] + tmp[1] + tmp[2] + base_addr;
          }
        }
      }
    }
    else if(array_info[0].distance > chunk_size){ // array_info[0].distance > chunk_size >= array_info[1].distance
      chunk_len = chunk_size / array_info[1].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j+=chunk_len){
          tmp[1] = stride_offset[1] * j;
          addrs[num++] = tmp[0] + tmp[1] + base_addr;
        }
      }
    }
    else{                                   // chunk_size >= array_info[0].distance
      chunk_len = chunk_size / array_info[0].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i+=chunk_len){
        addrs[num++] = stride_offset[0] * i + base_addr;
      }
    }
    break;
  case 4:
    if(array_info[2].distance > chunk_size){ // array_info[2].distance > chunk_size >= array_info[3].distance
      chunk_len = chunk_size / array_info[3].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(size_t k=0;k<array_info[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(size_t l=0;l<array_info[3].length;l+=chunk_len){
              tmp[3] = stride_offset[3] * l;
              addrs[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + base_addr;
            }
          }
        }
      }
    }
    else if(array_info[1].distance > chunk_size){ // array_info[1].distance > chunk_size >= array_info[2].distance
      chunk_len = chunk_size / array_info[2].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(size_t k=0;k<array_info[2].length;k+=chunk_len){
            tmp[2] = stride_offset[2] * k;
            addrs[num++] = tmp[0] + tmp[1] + tmp[2] + base_addr;
          }
        }
      }
    }
    else if(array_info[0].distance > chunk_size){ // array_info[0].distance > chunk_size >= array_info[1].distance
      chunk_len = chunk_size / array_info[1].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j+=chunk_len){
          tmp[1] = stride_offset[1] * j;
          addrs[num++] = tmp[0] + tmp[1] + base_addr;
        }
      }
    }
    else{                                   // chunk_size >= array_info[0].distance
      chunk_len = chunk_size / array_info[0].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i+=chunk_len){
        addrs[num++] = stride_offset[0] * i + base_addr;
      }
    }
    break;
  case 5:
    if(array_info[3].distance > chunk_size){ // array_info[3].distance > chunk_size >= array_info[4].distance
      chunk_len = chunk_size / array_info[4].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(size_t k=0;k<array_info[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(size_t l=0;l<array_info[3].length;l++){
              tmp[3] = stride_offset[3] * l;
              for(size_t m=0;m<array_info[4].length;m+=chunk_len){
                tmp[4] = stride_offset[4] * m;
                addrs[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + base_addr;
              }
            }
          }
        }
      }
    }
    else if(array_info[2].distance > chunk_size){ // array_info[2].distance > chunk_size >= array_info[3].distance
      chunk_len = chunk_size / array_info[3].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(size_t k=0;k<array_info[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(size_t l=0;l<array_info[3].length;l+=chunk_len){
              tmp[3] = stride_offset[3] * l;
              addrs[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + base_addr;
            }
          }
        }
      }
    }
    else if(array_info[1].distance > chunk_size){ // array_info[1].distance > chunk_size >= array_info[2].distance
      chunk_len = chunk_size / array_info[2].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(size_t k=0;k<array_info[2].length;k+=chunk_len){
            tmp[2] = stride_offset[2] * k;
            addrs[num++] = tmp[0] + tmp[1] + tmp[2] + base_addr;
          }
        }
      }
    }
    else if(array_info[0].distance > chunk_size){ // array_info[0].distance > chunk_size >= array_info[1].distance
      chunk_len = chunk_size / array_info[1].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j+=chunk_len){
          tmp[1] = stride_offset[1] * j;
          addrs[num++] = tmp[0] + tmp[1] + base_addr;
        }
      }
    }
    else{                                   // chunk_size >= array_info[0].distance
      chunk_len = chunk_size / array_info[0].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i+=chunk_len){
        addrs[num++] = stride_offset[0] * i + base_addr;
      }
    }
    break;
  case 6:
    if(array_info[4].distance > chunk_size){ // array_info[4].distance > chunk_size >= array_info[5].distance
      chunk_len = chunk_size / array_info[5].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(size_t k=0;k<array_info[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(size_t l=0;l<array_info[3].length;l++){
              tmp[3] = stride_offset[3] * l;
              for(size_t m=0;m<array_info[4].length;m++){
                tmp[4] = stride_offset[4] * m;
                for(size_t n=0;n<array_info[5].length;n+=chunk_len){
                  tmp[5] = stride_offset[5] * n;
                  addrs[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + base_addr;
                }
              }
            }
          }
        }
      }
    }
    else if(array_info[3].distance > chunk_size){ // array_info[3].distance > chunk_size >= array_info[4].distance
      chunk_len = chunk_size / array_info[4].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(size_t k=0;k<array_info[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(size_t l=0;l<array_info[3].length;l++){
              tmp[3] = stride_offset[3] * l;
              for(size_t m=0;m<array_info[4].length;m+=chunk_len){
                tmp[4] = stride_offset[4] * m;
                addrs[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + base_addr;
              }
            }
          }
        }
      }
    }
    else if(array_info[2].distance > chunk_size){ // array_info[2].distance > chunk_size >= array_info[3].distance
      chunk_len = chunk_size / array_info[3].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(size_t k=0;k<array_info[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(size_t l=0;l<array_info[3].length;l+=chunk_len){
              tmp[3] = stride_offset[3] * l;
              addrs[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + base_addr;
            }
          }
        }
      }
    }
    else if(array_info[1].distance > chunk_size){ // array_info[1].distance > chunk_size >= array_info[2].distance
      chunk_len = chunk_size / array_info[2].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(size_t k=0;k<array_info[2].length;k+=chunk_len){
            tmp[2] = stride_offset[2] * k;
            addrs[num++] = tmp[0] + tmp[1] + tmp[2] + base_addr;
          }
        }
      }
    }
    else if(array_info[0].distance > chunk_size){ // array_info[0].distance > chunk_size >= array_info[1].distance
      chunk_len = chunk_size / array_info[1].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j+=chunk_len){
          tmp[1] = stride_offset[1] * j;
          addrs[num++] = tmp[0] + tmp[1] + base_addr;
        }
      }
    }
    else{                                   // chunk_size >= array_info[0].distance
      chunk_len = chunk_size / array_info[0].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i+=chunk_len){
        addrs[num++] = stride_offset[0] * i + base_addr;
      }
    }
    break;
  case 7:
    if(array_info[5].distance > chunk_size){ // array_info[5].distance > chunk_size >= array_info[6].distance
      chunk_len = chunk_size / array_info[6].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(size_t k=0;k<array_info[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(size_t l=0;l<array_info[3].length;l++){
              tmp[3] = stride_offset[3] * l;
              for(size_t m=0;m<array_info[4].length;m++){
                tmp[4] = stride_offset[4] * m;
                for(size_t n=0;n<array_info[5].length;n++){
                  tmp[5] = stride_offset[5] * n;
                  for(size_t p=0;p<array_info[6].length;p+=chunk_len){
                    tmp[6] = stride_offset[6] * p;
                    addrs[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + base_addr;
                  }
                }
              }
            }
          }
        }
      }
    }
    else if(array_info[4].distance > chunk_size){ // array_info[4].distance > chunk_size >= array_info[5].distance
      chunk_len = chunk_size / array_info[5].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(size_t k=0;k<array_info[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(size_t l=0;l<array_info[3].length;l++){
              tmp[3] = stride_offset[3] * l;
              for(size_t m=0;m<array_info[4].length;m++){
                tmp[4] = stride_offset[4] * m;
                for(size_t n=0;n<array_info[5].length;n+=chunk_len){
                  tmp[5] = stride_offset[5] * n;
                  addrs[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + base_addr;
                }
              }
            }
          }
        }
      }
    }
    else if(array_info[3].distance > chunk_size){ // array_info[3].distance > chunk_size >= array_info[4].distance
      chunk_len = chunk_size / array_info[4].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(size_t k=0;k<array_info[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(size_t l=0;l<array_info[3].length;l++){
              tmp[3] = stride_offset[3] * l;
              for(size_t m=0;m<array_info[4].length;m+=chunk_len){
                tmp[4] = stride_offset[4] * m;
                addrs[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + base_addr;
              }
            }
          }
        }
      }
    }
    else if(array_info[2].distance > chunk_size){ // array_info[2].distance > chunk_size >= array_info[3].distance
      chunk_len = chunk_size / array_info[3].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(size_t k=0;k<array_info[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(size_t l=0;l<array_info[3].length;l+=chunk_len){
              tmp[3] = stride_offset[3] * l;
              addrs[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + base_addr;
            }
          }
        }
      }
    }
    else if(array_info[1].distance > chunk_size){ // array_info[1].distance > chunk_size >= array_info[2].distance
      chunk_len = chunk_size / array_info[2].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(size_t j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(size_t k=0;k<array_info[2].length;k+=chunk_len){
	    tmp[2] = stride_offset[2] * k;
	    addrs[num++] = tmp[0] + tmp[1] + tmp[2] + base_addr;
	  }
	}
      }
    }
    else if(array_info[0].distance > chunk_size){ // array_info[0].distance > chunk_size >= array_info[1].distance
      chunk_len = chunk_size / array_info[1].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j+=chunk_len){
          tmp[1] = stride_offset[1] * j;
          addrs[num++] = tmp[0] + tmp[1] + base_addr;
        }
      }
    }
    else{                                   // chunk_size >= array_info[0].distance
      chunk_len = chunk_size / array_info[0].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i+=chunk_len){
        addrs[num++] = stride_offset[0] * i + base_addr;
      }
    }
    break;
  }
}

static void _XMP_set_coarray_addresses(const uint64_t addr, const _XMP_array_section_t *array, const int dims, 
				       const size_t elmts, uint64_t* addrs)
{
  uint64_t stride_offset[dims], tmp[dims];

  // Temporally variables to reduce calculation for offset
  for(int i=0;i<dims;i++)
    stride_offset[i] = array[i].stride * array[i].distance;
 
  switch (dims){
  case 1:
    for(size_t i=0, num=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      addrs[num++] = addr + tmp[0];
    }
    break;
  case 2:
    for(size_t i=0, num=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(size_t j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
	addrs[num++] = addr + tmp[0] + tmp[1];
      }
    }
    break;
  case 3:
    for(size_t i=0, num=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(size_t j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
	for(size_t k=0;k<array[2].length;k++){
	  tmp[2] = stride_offset[2] * k;
	  addrs[num++] = addr + tmp[0] + tmp[1] + tmp[2];
	}
      }
    }
    break;
  case 4:
    for(size_t i=0, num=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(size_t j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
	for(size_t k=0;k<array[2].length;k++){
          tmp[2] = stride_offset[2] * k;
	  for(size_t l=0;l<array[3].length;l++){
	    tmp[3] = stride_offset[3] * l;
	    addrs[num++] = addr + tmp[0] + tmp[1] + tmp[2] + tmp[3];
	  }
	}
      }
    }
    break;
  case 5:
    for(size_t i=0, num=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(size_t j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(size_t k=0;k<array[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(size_t l=0;l<array[3].length;l++){
            tmp[3] = stride_offset[3] * l;
	    for(size_t m=0;m<array[4].length;m++){
	      tmp[4] = stride_offset[4] * m;
	      addrs[num++] = addr + tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4];
	    }
          }
        }
      }
    }
    break;
  case 6:
    for(size_t i=0, num=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(size_t j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(size_t k=0;k<array[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(size_t l=0;l<array[3].length;l++){
            tmp[3] = stride_offset[3] * l;
            for(size_t m=0;m<array[4].length;m++){
              tmp[4] = stride_offset[4] * m;
	      for(size_t n=0;n<array[5].length;n++){
		tmp[5] = stride_offset[5] * n;
		addrs[num++] = addr + tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5];
	      }
            }
          }
        }
      }
    }
    break;
  case 7:
    for(size_t i=0, num=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(size_t j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(size_t k=0;k<array[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(size_t l=0;l<array[3].length;l++){
            tmp[3] = stride_offset[3] * l;
            for(size_t m=0;m<array[4].length;m++){
              tmp[4] = stride_offset[4] * m;
              for(size_t n=0;n<array[5].length;n++){
                tmp[5] = stride_offset[5] * n;
		for(size_t p=0;p<array[6].length;p++){
		  tmp[6] = stride_offset[6] * p;
		  addrs[num++] = addr + tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6];
		}
	      }
            }
          }
        }
      }
    }
    break;
  }
}

/***************************************************************************/
/* DESCRIPTION : Check the dimension of an array has all element ?         */
/* ARGUMENT    : [IN] *array_info : Information of array                   */
/*               [IN] dim         : Dimension                              */
/* RETURN      : If the dimension of an array has all element, return TRUE */
/***************************************************************************/
static int _is_all_element(const _XMP_array_section_t *array_info, int dim){
  if(array_info[dim].elmts == array_info[dim].length)
    return _XMP_N_INT_TRUE;
  else
    return _XMP_N_INT_FALSE;
}


/**************************************************************************************/
/* DESCRIPTION : Check round of array[dim]                                            */
/* ARGUMENT    : [IN] *array      : Information of array                              */
/*               [IN] dim         : Dimension                                         */
/* RETURN      : If a round of array[dim], return TRUE                                */
/* NOTE        : The following 3 lines are equal to this function                     */
/*    int last_elmt = array[dim].start + (array[dim].length - 1) * array[dim].stride; */
/*    int next_elmt = last_elmt + array[dim].stride - array[dim].elmts;               */
/*    return next_elmt == array[dim].start;                                           */
/**************************************************************************************/
static int _check_round(const _XMP_array_section_t *array, const int dim)
{
  return array[dim].length * array[dim].stride - array[dim].elmts == 0;
}

/**
   If 1dim array has a constant stride, return TRUE (Always TRUE)
*/
static int _is_constant_stride_1dim()
{
  return _XMP_N_INT_TRUE;
}

/********************************************************************/
/* DESCRIPTION : Is 2dim array has a constant stride ?              */
/* ARGUMENT    : [IN] *array_info : Information of array            */
/* RETURN:     : If 2dim array has a constant stride, return TRUE   */
/********************************************************************/
static int _is_constant_stride_2dim(const _XMP_array_section_t *array_info)
{
  if(array_info[0].stride == 1 && _check_round(array_info, 1)){
    return _XMP_N_INT_TRUE;
  }
  else if(array_info[1].stride == 1){
    return _XMP_N_INT_TRUE;
  }

  return _XMP_N_INT_FALSE;
}

/********************************************************************/
/* DESCRIPTION : Is 3dim array has a constant stride ?              */
/* ARGUMENT    : [IN] *array_info : Information of array            */
/* RETURN:     : If 3dim array has a constant stride, return TRUE   */
/********************************************************************/
static int _is_constant_stride_3dim(const _XMP_array_section_t *array_info)
{
  if(array_info[1].stride == 1 && _is_all_element(array_info, 2)){
    return _XMP_N_INT_TRUE;
  }
  else if(array_info[0].stride == 1){
    if(_check_round(array_info, 1) && array_info[2].stride == 1){
      return _XMP_N_INT_TRUE;
    }
    else if(_is_all_element(array_info, 1) && _check_round(array_info, 2)){
      return _XMP_N_INT_TRUE;
    }
  }

  return _XMP_N_INT_FALSE;
}

/********************************************************************/
/* DESCRIPTION : Is 4dim array has a constant stride ?              */
/* ARGUMENT    : [IN] *array_info : Information of array            */
/* RETURN:     : If 4dim array has a constant stride, return TRUE   */
/********************************************************************/
static int _is_constant_stride_4dim(const _XMP_array_section_t *array_info)
{
  if(array_info[1].stride == 1 && _is_all_element(array_info, 2) &&
     _is_all_element(array_info, 3)){
    return _XMP_N_INT_TRUE;
  }
  else if(array_info[0].stride == 1){
    if(_check_round(array_info, 1) && array_info[2].stride == 1 &&
       _is_all_element(array_info, 3)){
      return _XMP_N_INT_TRUE;
    }
    else if(_is_all_element(array_info, 1) && _check_round(array_info, 2) &&
            array_info[3].stride == 1){
      return _XMP_N_INT_TRUE;
    }
    else if(_is_all_element(array_info, 1) && _is_all_element(array_info, 2) &&
            _check_round(array_info, 3)){
      return _XMP_N_INT_TRUE;
    }
  }

  return _XMP_N_INT_FALSE;
}

/********************************************************************/
/* DESCRIPTION : Is 5dim array has a constant stride ?              */
/* ARGUMENT    : [IN] *array_info : Information of array            */
/* RETURN:     : If 5dim array has a constant stride, return TRUE   */
/********************************************************************/
static int _is_constant_stride_5dim(const _XMP_array_section_t *array_info)
{
  if(array_info[1].stride == 1 && _is_all_element(array_info, 2) &&
     _is_all_element(array_info, 3) && _is_all_element(array_info, 4)){
    return _XMP_N_INT_TRUE;
  }
  else if(array_info[0].stride == 1){
    if(_check_round(array_info, 1) && array_info[2].stride == 1 &&
       _is_all_element(array_info, 3) && _is_all_element(array_info, 4)){
      return _XMP_N_INT_TRUE;
    }
    else if(_is_all_element(array_info, 1) && _check_round(array_info, 2) &&
            array_info[3].stride == 1 && _is_all_element(array_info, 4)){
      return _XMP_N_INT_TRUE;
    }
    else if(_is_all_element(array_info, 1) && _is_all_element(array_info, 2) &&
            _check_round(array_info, 3) && array_info[4].stride == 1){
      return _XMP_N_INT_TRUE;
    }
    else if(_is_all_element(array_info, 1) && _is_all_element(array_info, 2) &&
            _is_all_element(array_info, 3) && _check_round(array_info, 4)){
      return _XMP_N_INT_TRUE;
    }
  }

  return _XMP_N_INT_FALSE;
}

/********************************************************************/
/* DESCRIPTION : Is 6dim array has a constant stride ?              */
/* ARGUMENT    : [IN] *array_info : Information of array            */
/* RETURN:     : If 6dim array has a constant stride, return TRUE   */
/********************************************************************/
static int _is_constant_stride_6dim(const _XMP_array_section_t *array_info)
{
  if(array_info[1].stride == 1 && _is_all_element(array_info, 2) &&
     _is_all_element(array_info, 3) && _is_all_element(array_info, 4) &&
     _is_all_element(array_info, 5)){
    return _XMP_N_INT_TRUE;
  }
  else if(array_info[0].stride == 1){
    if(_check_round(array_info, 1) && array_info[2].stride == 1 &&
       _is_all_element(array_info, 3) && _is_all_element(array_info, 4) &&
       _is_all_element(array_info, 5)){
      return _XMP_N_INT_TRUE;
    }
    else if(_is_all_element(array_info, 1) && _check_round(array_info, 2) &&
            array_info[3].stride == 1 && _is_all_element(array_info, 4) &&
            _is_all_element(array_info, 5)){
      return _XMP_N_INT_TRUE;
    }
    else if(_is_all_element(array_info, 1) && _is_all_element(array_info, 2) &&
            _check_round(array_info, 3) && array_info[4].stride == 1 &&
            _is_all_element(array_info, 5)){
      return _XMP_N_INT_TRUE;
    }
    else if(_is_all_element(array_info, 1) && _is_all_element(array_info, 2) &&
            _is_all_element(array_info, 3) && _check_round(array_info, 4) &&
            array_info[5].stride == 1){
      return _XMP_N_INT_TRUE;
    }
    else if(_is_all_element(array_info, 1) && _is_all_element(array_info, 2) &&
            _is_all_element(array_info, 3) && _is_all_element(array_info, 4) &&
            _check_round(array_info, 5)){
      return _XMP_N_INT_TRUE;
    }
  }

  return _XMP_N_INT_FALSE;
}

/********************************************************************/
/* DESCRIPTION : Is 7dim array has a constant stride ?              */
/* ARGUMENT    : [IN] *array_info : Information of array            */
/* RETURN:     : If 7dim array has a constant stride, return TRUE   */
/********************************************************************/
static int _is_constant_stride_7dim(const _XMP_array_section_t *array_info)
{
  if(array_info[1].stride == 1 && _is_all_element(array_info, 2) &&
     _is_all_element(array_info, 3) && _is_all_element(array_info, 4) &&
     _is_all_element(array_info, 5) && _is_all_element(array_info, 6)){
    return _XMP_N_INT_TRUE;
  }
  else if(array_info[0].stride == 1){
    if(_check_round(array_info, 1) && array_info[2].stride == 1 &&
       _is_all_element(array_info, 3) && _is_all_element(array_info, 4) &&
       _is_all_element(array_info, 5) && _is_all_element(array_info, 6)){
      return _XMP_N_INT_TRUE;
    }
    else if(_is_all_element(array_info, 1) && _check_round(array_info, 2) &&
            array_info[3].stride == 1 && _is_all_element(array_info, 4) &&
            _is_all_element(array_info, 5) && _is_all_element(array_info, 6)){
      return _XMP_N_INT_TRUE;
    }
    else if(_is_all_element(array_info, 1) && _is_all_element(array_info, 2) &&
            _check_round(array_info, 3) && array_info[4].stride == 1 &&
            _is_all_element(array_info, 5) && _is_all_element(array_info, 6)){
      return _XMP_N_INT_TRUE;
    }
    else if(_is_all_element(array_info, 1) && _is_all_element(array_info, 2) &&
            _is_all_element(array_info, 3) && _check_round(array_info, 4) &&
            array_info[5].stride == 1 && _is_all_element(array_info, 6)){
      return _XMP_N_INT_TRUE;
    }
    else if(_is_all_element(array_info, 1) && _is_all_element(array_info, 2) &&
            _is_all_element(array_info, 3) && _is_all_element(array_info, 4) &&
            _check_round(array_info, 5) && array_info[6].stride == 1){
      return _XMP_N_INT_TRUE;
    }
    else if(_is_all_element(array_info, 1) && _is_all_element(array_info, 2) &&
            _is_all_element(array_info, 3) && _is_all_element(array_info, 4) &&
            _is_all_element(array_info, 5) && _check_round(array_info, 6)){
      return _XMP_N_INT_TRUE;
    }
  }

  return _XMP_N_INT_FALSE;
}

/**********************************************************************************/
/* DESCRIPTION : Check shape of two arrays, the same is except for start          */
/* ARGUMENT    : [IN] *array1_info : Information of array1                        */
/*               [IN] *array2_info : Information of array2                        */
/*               [IN] array1_dims  : Number of dimensions of array1               */
/*               [IN] array2_dims  : Number of dimensions of array2               */
/* RETURN:     : If two arrays have the same stride except for start, return TRUE */
/**********************************************************************************/
static int _is_the_same_shape_except_for_start(const _XMP_array_section_t *array1_info,
                                               const _XMP_array_section_t *array2_info,
                                               const int array1_dims, const int array2_dims)
{
  if(array1_dims != array2_dims) return _XMP_N_INT_FALSE;

  for(int i=0;i<array1_dims;i++)
    if(array1_info[i].length != array2_info[i].length ||
       array1_info[i].elmts  != array2_info[i].elmts ||
       array1_info[i].stride != array2_info[i].stride)
      return _XMP_N_INT_FALSE;

  return _XMP_N_INT_TRUE;
}

/********************************************************************/
/* DESCRIPTION : Check two arrays have the same stride              */
/* ARGUMENT    : [IN] *array1_info : Information of array1          */
/*               [IN] *array2_info : Information of array2          */
/*               [IN] array1_dims  : Number of dimensions of array1 */
/*               [IN] array2_dims  : Number of dimensions of array2 */
/* RETURN:     : If two arrays have the same stride, return TRUE    */
/* NOTE        : This function does not support the following very  */
/*               rare case.                                         */
/*               int a[10][10]; -> a[0:2][0:5:2];                   */
/*               An array has continuity jumped over the dimension  */
/********************************************************************/
static int _XMP_is_the_same_constant_stride(const _XMP_array_section_t *array1_info,
					    const _XMP_array_section_t *array2_info,
					    const int array1_dims, const int array2_dims)
{
  if(! _is_the_same_shape_except_for_start(array1_info, array2_info,
                                           array1_dims, array2_dims))
    return _XMP_N_INT_FALSE;

  switch (array1_dims){
  case 1:
    return _is_constant_stride_1dim();
  case 2:
    return _is_constant_stride_2dim(array1_info);
  case 3:
    return _is_constant_stride_3dim(array1_info);
  case 4:
    return _is_constant_stride_4dim(array1_info);
  case 5:
    return _is_constant_stride_5dim(array1_info);
  case 6:
    return _is_constant_stride_6dim(array1_info);
  case 7:
    return _is_constant_stride_7dim(array1_info);
  default:
    _XMP_fatal("Coarray Error ! Dimension is too big.\n");
    return _XMP_N_INT_FALSE; // dummy
  }
}

/***************************************************************/
/* DESCRIPTION : Caluculate stride size of array               */
/* ARGUMENT    : [IN] *array_info : Information of array       */
/*               [IN] dims        : Demension of array         */
/*               [IN] chunk_size  : Size of chunk              */
/* RETURN:     : Stride size                                   */
/***************************************************************/
static size_t _XMP_calc_stride(const _XMP_array_section_t *array_info, const int dims,
			       const size_t chunk_size)
{
  uint64_t stride_offset[dims], tmp[dims];
  size_t stride[2];

  // Temporally variables to reduce calculation for offset
  for(int i=0;i<dims;i++)
    stride_offset[i] = array_info[i].stride * array_info[i].distance;

  switch (dims){
    size_t chunk_len;
  case 1:
    chunk_len = chunk_size / array_info[0].distance;
    for(size_t i=0,num=0;i<array_info[0].length;i+=chunk_len){
      stride[num++] = stride_offset[0] * chunk_len * i;
      if(num == 2) goto end;
    }
  case 2:
    if(array_info[0].distance > chunk_size){ // array_info[0].distance > chunk_size >= array_info[1].distance
      chunk_len = chunk_size / array_info[1].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j+=chunk_len){
          tmp[1] = stride_offset[1] * j;
          stride[num++] = tmp[0] + tmp[1];
          if(num == 2) goto end;
        }
      }
    }
    else{                               // chunk_size >= array_info[0].distance
      chunk_len = chunk_size / array_info[0].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i+=chunk_len){
        stride[num++] = stride_offset[0] * i;
        if(num == 2) goto end;
      }
    }
  case 3:
   if(array_info[1].distance > chunk_size){ // array_info[1].distance > chunk_size >= array_info[2].distance
     chunk_len = chunk_size / array_info[2].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
       tmp[0] = stride_offset[0] * i;
       for(size_t j=0;j<array_info[1].length;j++){
	 tmp[1] = stride_offset[1] * j;
	 for(size_t k=0;k<array_info[2].length;k+=chunk_len){
	   tmp[2] = stride_offset[2] * k;
	   stride[num++] = tmp[0] + tmp[1] + tmp[2];
	   if(num == 2) goto end;
	 }
       }
     }
   }
   else if(array_info[0].distance > chunk_size){ // array_info[0].distance > chunk_size >= array_info[1].distance
     chunk_len = chunk_size / array_info[1].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
       tmp[0] = stride_offset[0] * i;
       for(size_t j=0;j<array_info[1].length;j+=chunk_len){
	 tmp[1] = stride_offset[1] * j;
	 stride[num++] = tmp[0] + tmp[1];
	 if(num == 2) goto end;
       }
     }
   }
   else{                                   // chunk_size >= array_info[0].distance
     chunk_len = chunk_size / array_info[0].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i+=chunk_len){
       stride[num++] = stride_offset[0] * i;
       if(num == 2) goto end;
     }
   }
   break;
 case 4:
   if(array_info[2].distance > chunk_size){ // array_info[2].distance > chunk_size >= array_info[3].distance
     chunk_len = chunk_size / array_info[3].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
       tmp[0] = stride_offset[0] * i;
       for(size_t j=0;j<array_info[1].length;j++){
	 tmp[1] = stride_offset[1] * j;
	 for(size_t k=0;k<array_info[2].length;k++){
	   tmp[2] = stride_offset[2] * k;
	   for(size_t l=0;l<array_info[3].length;l+=chunk_len){
	     tmp[3] = stride_offset[3] * l;
	     stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
	     if(num == 2) goto end;
	   }
	 }
       }
     }
   }
   else if(array_info[1].distance > chunk_size){ // array_info[1].distance > chunk_size >= array_info[2].distance
     chunk_len = chunk_size / array_info[2].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
       tmp[0] = stride_offset[0] * i;
       for(size_t j=0;j<array_info[1].length;j++){
	 tmp[1] = stride_offset[1] * j;
	 for(size_t k=0;k<array_info[2].length;k+=chunk_len){
	   tmp[2] = stride_offset[2] * k;
	   stride[num++] = tmp[0] + tmp[1] + tmp[2];
	   if(num == 2) goto end;
	 }
       }
     }
   }
   else if(array_info[0].distance > chunk_size){ // array_info[0].distance > chunk_size >= array_info[1].distance
     chunk_len = chunk_size / array_info[1].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
       tmp[0] = stride_offset[0] * i;
       for(size_t j=0;j<array_info[1].length;j+=chunk_len){
	 tmp[1] = stride_offset[1] * j;
	 stride[num++] = tmp[0] + tmp[1];
	 if(num == 2) goto end;
       }
     }
   }
   else{                                   // chunk_size >= array_info[0].distance
     chunk_len = chunk_size / array_info[0].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i+=chunk_len){
       stride[num++] = stride_offset[0] * i;
       if(num == 2) goto end;
     }
   }
   break;
 case 5:
   if(array_info[3].distance > chunk_size){ // array_info[3].distance > chunk_size >= array_info[4].distance
     chunk_len = chunk_size / array_info[4].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
       tmp[0] = stride_offset[0] * i;
       for(size_t j=0;j<array_info[1].length;j++){
	 tmp[1] = stride_offset[1] * j;
	 for(size_t k=0;k<array_info[2].length;k++){
	   tmp[2] = stride_offset[2] * k;
	   for(size_t l=0;l<array_info[3].length;l++){
	     tmp[3] = stride_offset[3] * l;
	     for(size_t m=0;m<array_info[4].length;m+=chunk_len){
	       tmp[4] = stride_offset[4] * m;
	       stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4];
	       if(num == 2) goto end;
	     }
	   }
	 }
       }
     }
   }
   else if(array_info[2].distance > chunk_size){ // array_info[2].distance > chunk_size >= array_info[3].distance
     chunk_len = chunk_size / array_info[3].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
       tmp[0] = stride_offset[0] * i;
       for(size_t j=0;j<array_info[1].length;j++){
	 tmp[1] = stride_offset[1] * j;
	 for(size_t k=0;k<array_info[2].length;k++){
	   tmp[2] = stride_offset[2] * k;
	   for(size_t l=0;l<array_info[3].length;l+=chunk_len){
	     tmp[3] = stride_offset[3] * l;
	     stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
	     if(num == 2) goto end;
	   }
	 }
       }
     }
   }
   else if(array_info[1].distance > chunk_size){ // array_info[1].distance > chunk_size >= array_info[2].distance
     chunk_len = chunk_size / array_info[2].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
       tmp[0] = stride_offset[0] * i;
       for(size_t j=0;j<array_info[1].length;j++){
	 tmp[1] = stride_offset[1] * j;
	 for(size_t k=0;k<array_info[2].length;k+=chunk_len){
	   tmp[2] = stride_offset[2] * k;
	   stride[num++] = tmp[0] + tmp[1] + tmp[2];
	   if(num == 2) goto end;
	 }
       }
     }
   }
   else if(array_info[0].distance > chunk_size){ // array_info[0].distance > chunk_size >= array_info[1].distance
     chunk_len = chunk_size / array_info[1].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
       tmp[0] = stride_offset[0] * i;
       for(size_t j=0;j<array_info[1].length;j+=chunk_len){
	 tmp[1] = stride_offset[1] * j;
	 stride[num++] = tmp[0] + tmp[1];
	 if(num == 2) goto end;
       }
     }
   }
   else{                                   // chunk_size >= array_info[0].distance
     chunk_len = chunk_size / array_info[0].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i+=chunk_len){
       stride[num++] = stride_offset[0] * i;
       if(num == 2) goto end;
     }
   }
   break;
 case 6:
   if(array_info[4].distance > chunk_size){ // array_info[4].distance > chunk_size >= array_info[5].distance
     chunk_len = chunk_size / array_info[5].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
       tmp[0] = stride_offset[0] * i;
       for(size_t j=0;j<array_info[1].length;j++){
	 tmp[1] = stride_offset[1] * j;
	 for(size_t k=0;k<array_info[2].length;k++){
	   tmp[2] = stride_offset[2] * k;
	   for(size_t l=0;l<array_info[3].length;l++){
	     tmp[3] = stride_offset[3] * l;
	     for(size_t m=0;m<array_info[4].length;m++){
	       tmp[4] = stride_offset[4] * m;
	       for(size_t n=0;n<array_info[5].length;n+=chunk_len){
		 tmp[5] = stride_offset[5] * n;
		 stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5];
		 if(num == 2) goto end;
	       }
	     }
	   }
	 }
       }
     }
   }
   else if(array_info[3].distance > chunk_size){ // array_info[3].distance > chunk_size >= array_info[4].distance
     chunk_len = chunk_size / array_info[4].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
       tmp[0] = stride_offset[0] * i;
       for(size_t j=0;j<array_info[1].length;j++){
	 tmp[1] = stride_offset[1] * j;
	 for(size_t k=0;k<array_info[2].length;k++){
	   tmp[2] = stride_offset[2] * k;
	   for(size_t l=0;l<array_info[3].length;l++){
	     tmp[3] = stride_offset[3] * l;
	     for(size_t m=0;m<array_info[4].length;m+=chunk_len){
	       tmp[4] = stride_offset[4] * m;
	       stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4];
               if(num == 2) goto end;
	     }
	   }
	 }
       }
     }
   }
   else if(array_info[2].distance > chunk_size){ // array_info[2].distance > chunk_size >= array_info[3].distance
     chunk_len = chunk_size / array_info[3].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
       tmp[0] = stride_offset[0] * i;
       for(size_t j=0;j<array_info[1].length;j++){
	 tmp[1] = stride_offset[1] * j;
	 for(size_t k=0;k<array_info[2].length;k++){
	   tmp[2] = stride_offset[2] * k;
	   for(size_t l=0;l<array_info[3].length;l+=chunk_len){
	     tmp[3] = stride_offset[3] * l;
	     stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
	     if(num == 2) goto end;
	   }
	 }
       }
     }
   }
   else if(array_info[1].distance > chunk_size){ // array_info[1].distance > chunk_size >= array_info[2].distance
     chunk_len = chunk_size / array_info[2].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
       tmp[0] = stride_offset[0] * i;
       for(size_t j=0;j<array_info[1].length;j++){
	 tmp[1] = stride_offset[1] * j;
	 for(size_t k=0;k<array_info[2].length;k+=chunk_len){
	   tmp[2] = stride_offset[2] * k;
	   stride[num++] = tmp[0] + tmp[1] + tmp[2];
	   if(num == 2) goto end;
	 }
       }
     }
   }
   else if(array_info[0].distance > chunk_size){ // array_info[0].distance > chunk_size >= array_info[1].distance
     chunk_len = chunk_size / array_info[1].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
       tmp[0] = stride_offset[0] * i;
       for(size_t j=0;j<array_info[1].length;j+=chunk_len){
	 tmp[1] = stride_offset[1] * j;
	 stride[num++] = tmp[0] + tmp[1];
	 if(num == 2) goto end;
       }
     }
   }
   else{                                   // chunk_size >= array_info[0].distance
     chunk_len = chunk_size / array_info[0].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i+=chunk_len){
       stride[num++] = stride_offset[0] * i;
       if(num == 2) goto end;
     }
   }
   break;
 case 7:
   if(array_info[5].distance > chunk_size){ // array_info[5].distance > chunk_size >= array_info[6].distance
     chunk_len = chunk_size / array_info[6].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
       tmp[0] = stride_offset[0] * i;
       for(size_t j=0;j<array_info[1].length;j++){
	 tmp[1] = stride_offset[1] * j;
	 for(size_t k=0;k<array_info[2].length;k++){
	   tmp[2] = stride_offset[2] * k;
	   for(size_t l=0;l<array_info[3].length;l++){
	     tmp[3] = stride_offset[3] * l;
	     for(size_t m=0;m<array_info[4].length;m++){
	       tmp[4] = stride_offset[4] * m;
	       for(size_t n=0;n<array_info[5].length;n++){
		 tmp[5] = stride_offset[5] * n;
		 for(size_t p=0;p<array_info[6].length;p+=chunk_len){
		   tmp[6] = stride_offset[6] * p;
		   stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6];
		   if(num == 2) goto end;
		 }
	       }
	     }
	   }
	 }
       }
     }
   }
   else if(array_info[4].distance > chunk_size){ // array_info[4].distance > chunk_size >= array_info[5].distance
     chunk_len = chunk_size / array_info[5].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
       tmp[0] = stride_offset[0] * i;
       for(size_t j=0;j<array_info[1].length;j++){
	 tmp[1] = stride_offset[1] * j;
	 for(size_t k=0;k<array_info[2].length;k++){
	   tmp[2] = stride_offset[2] * k;
	   for(size_t l=0;l<array_info[3].length;l++){
	     tmp[3] = stride_offset[3] * l;
	     for(size_t m=0;m<array_info[4].length;m++){
	       tmp[4] = stride_offset[4] * m;
	       for(size_t n=0;n<array_info[5].length;n+=chunk_len){
		 tmp[5] = stride_offset[5] * n;
		 stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5];
		 if(num == 2) goto end;
	       }
	     }
	   }
	 }
       }
     }
   }
   else if(array_info[3].distance > chunk_size){ // array_info[3].distance > chunk_size >= array_info[4].distance
     chunk_len = chunk_size / array_info[4].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
       tmp[0] = stride_offset[0] * i;
       for(size_t j=0;j<array_info[1].length;j++){
	 tmp[1] = stride_offset[1] * j;
	 for(size_t k=0;k<array_info[2].length;k++){
	   tmp[2] = stride_offset[2] * k;
	   for(size_t l=0;l<array_info[3].length;l++){
	     tmp[3] = stride_offset[3] * l;
	     for(size_t m=0;m<array_info[4].length;m+=chunk_len){
	       tmp[4] = stride_offset[4] * m;
	       stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4];
               if(num == 2) goto end;
	     }
	   }
	 }
       }
     }
   }
   else if(array_info[2].distance > chunk_size){ // array_info[2].distance > chunk_size >= array_info[3].distance
     chunk_len = chunk_size / array_info[3].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
       tmp[0] = stride_offset[0] * i;
       for(size_t j=0;j<array_info[1].length;j++){
	 tmp[1] = stride_offset[1] * j;
	 for(size_t k=0;k<array_info[2].length;k++){
	   tmp[2] = stride_offset[2] * k;
	   for(size_t l=0;l<array_info[3].length;l+=chunk_len){
	     tmp[3] = stride_offset[3] * l;
	     stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
	     if(num == 2) goto end;
	   }
	 }
       }
     }
   }
   else if(array_info[1].distance > chunk_size){ // array_info[1].distance > chunk_size >= array_info[2].distance
     chunk_len = chunk_size / array_info[2].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
       tmp[0] = stride_offset[0] * i;
       for(size_t j=0;j<array_info[1].length;j++){
	 tmp[1] = stride_offset[1] * j;
	 for(size_t k=0;k<array_info[2].length;k+=chunk_len){
	   tmp[2] = stride_offset[2] * k;
	   stride[num++] = tmp[0] + tmp[1] + tmp[2];
	   if(num == 2) goto end;
	 }
       }
     }
   }
   else if(array_info[0].distance > chunk_size){ // array_info[0].distance > chunk_size >= array_info[1].distance
     chunk_len = chunk_size / array_info[1].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
       tmp[0] = stride_offset[0] * i;
       for(size_t j=0;j<array_info[1].length;j+=chunk_len){
	 tmp[1] = stride_offset[1] * j;
	 stride[num++] = tmp[0] + tmp[1];
	 if(num == 2) goto end;
       }
     }
   }
   else{                                   // chunk_size >= array_info[0].distance
     chunk_len = chunk_size / array_info[0].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i+=chunk_len){
       stride[num++] = stride_offset[0] * i;
       if(num == 2) goto end;
     }
   }
   break;
  }

 end:
  return stride[1] - stride[0];
}
/**
   Execute sync_memory for put operation
 */
void _XMP_fjrdma_sync_memory_put()
{
  if(_XMP_flag_put_nb_rr){
    while(1){
      if(FJMPI_Rdma_poll_cq(_XMP_COARRAY_SEND_NIC_TMP_0, NULL) == FJMPI_RDMA_NOTICE)
	_num_of_puts--;
      if(_num_of_puts == 0) break;

      if(FJMPI_Rdma_poll_cq(_XMP_COARRAY_SEND_NIC_TMP_1, NULL) == FJMPI_RDMA_NOTICE)
	_num_of_puts--;
      if(_num_of_puts == 0) break;

      if(FJMPI_Rdma_poll_cq(_XMP_COARRAY_SEND_NIC_TMP_2, NULL) == FJMPI_RDMA_NOTICE)
	_num_of_puts--;
      if(_num_of_puts == 0) break;

      if(FJMPI_Rdma_poll_cq(_XMP_COARRAY_SEND_NIC_TMP_3, NULL) == FJMPI_RDMA_NOTICE)
	_num_of_puts--;
      if(_num_of_puts == 0) break;
    }
  }
  else{
    while(_num_of_puts != 0)
      if(FJMPI_Rdma_poll_cq(_XMP_COARRAY_SEND_NIC, NULL) == FJMPI_RDMA_NOTICE)
	_num_of_puts--;
  }
}

/**
   Execute sync_memory for get operation
*/
void _XMP_fjrdma_sync_memory_get()
{
  while(_num_of_gets != 0)
    if(FJMPI_Rdma_poll_cq(_XMP_COARRAY_SEND_NIC, NULL) == FJMPI_RDMA_NOTICE)
      _num_of_gets--;
}

/**
   Add 1 to _num_of_puts.
 */
void _XMP_add_num_of_puts()
{
  _num_of_puts++;
  if(_num_of_puts > _XMP_FJRDMA_MAX_PUT)
    _XMP_fjrdma_sync_memory_put();
}

/**
   Add 1 to _num_of_gets.
*/
void _XMP_add_num_of_gets()
{
  _num_of_gets++;
  if(_num_of_gets > _XMP_FJRDMA_MAX_GET)
    _XMP_fjrdma_sync_memory_get();
}

/**
   Execute sync_memory
*/
void _XMP_fjrdma_sync_memory()
{
  if(_XMP_flag_put_nb)
    _XMP_fjrdma_sync_memory_put();

  //  _XMP_fjrdma_sync_memory_put();
  // _XMP_fjrdma_sync_memory_get don't need to be executed
}

/**
   Execute sync_all
*/
void _XMP_fjrdma_sync_all()
{
  if(_XMP_flag_put_nb)
    _XMP_fjrdma_sync_memory();

  //  _XMP_fjrdma_sync_memory();
  MPI_Barrier(MPI_COMM_WORLD);
}

/**
   transfer_size must be 4-Byte align
*/
static void _check_transfer_size(const size_t transfer_size)
{
  if((transfer_size&0x3) != 0){  // transfer_size % 4 != 0
    fprintf(stderr, "transfer_size must be multiples of four (%zu)\n", transfer_size);
    exit(1);
  }
}

/**
   The _XMP_FJMPI_Rdma_put() is a wrapper function of the FJMPI_Rdma_put().
   FJMPI_Rdma_put() cannot transfer more than _XMP_FJRDMA_MAX_SIZE(about 16MB) data.
   Thus, the _XMP_FJMPI_Rdma_put() executes mutiple put operations to trasfer more than 16MB data.
*/
static void _XMP_FJMPI_Rdma_put(const int target_rank, uint64_t raddr, uint64_t laddr, 
				const size_t transfer_size)
{
  if(transfer_size <= _XMP_FJRDMA_MAX_SIZE){
    if(_XMP_flag_put_nb_rr_i){
      switch(_num_of_puts%4){
      case 0:
	FJMPI_Rdma_put(target_rank, _XMP_FJRDMA_TAG, raddr, laddr, transfer_size, _XMP_COARRAY_FLAG_NIC_TMP_i0);
        break;
      case 1:
	FJMPI_Rdma_put(target_rank, _XMP_FJRDMA_TAG, raddr, laddr, transfer_size, _XMP_COARRAY_FLAG_NIC_TMP_i1);
        break;
      case 2:
	FJMPI_Rdma_put(target_rank, _XMP_FJRDMA_TAG, raddr, laddr, transfer_size, _XMP_COARRAY_FLAG_NIC_TMP_i2);
        break;
      case 3:
	FJMPI_Rdma_put(target_rank, _XMP_FJRDMA_TAG, raddr, laddr, transfer_size, _XMP_COARRAY_FLAG_NIC_TMP_i3);
        break;
      default:
        printf("ERROR !! \n"); exit(1);
	break;
      }
    }
    else if(_XMP_flag_put_nb_rr){
      switch(_num_of_puts%4){
      case 0:
	FJMPI_Rdma_put(target_rank, _XMP_FJRDMA_TAG, raddr, laddr, transfer_size, _XMP_COARRAY_FLAG_NIC_TMP_0);
	break;
      case 1:
	FJMPI_Rdma_put(target_rank, _XMP_FJRDMA_TAG, raddr, laddr, transfer_size, _XMP_COARRAY_FLAG_NIC_TMP_1);
	break;
      case 2:
	FJMPI_Rdma_put(target_rank, _XMP_FJRDMA_TAG, raddr, laddr, transfer_size, _XMP_COARRAY_FLAG_NIC_TMP_2);
	break;
      case 3:
	FJMPI_Rdma_put(target_rank, _XMP_FJRDMA_TAG, raddr, laddr, transfer_size, _XMP_COARRAY_FLAG_NIC_TMP_3);
	break;
      default:
	printf("ERROR !! \n"); exit(1);
	break;
      }
    }
    else{
      FJMPI_Rdma_put(target_rank, _XMP_FJRDMA_TAG, raddr, laddr, transfer_size, _XMP_COARRAY_FLAG_NIC);
    }
    _XMP_add_num_of_puts();
  }
  else{
    int times = transfer_size / _XMP_FJRDMA_MAX_SIZE;
    int rest  = transfer_size - _XMP_FJRDMA_MAX_SIZE * times;

    for(int i=0;i<times;i++){
      FJMPI_Rdma_put(target_rank, _XMP_FJRDMA_TAG, raddr, laddr, _XMP_FJRDMA_MAX_SIZE, _XMP_COARRAY_FLAG_NIC);
      raddr += _XMP_FJRDMA_MAX_SIZE;
      laddr += _XMP_FJRDMA_MAX_SIZE;
      _XMP_add_num_of_puts();
    }
    
    if(rest != 0){
      FJMPI_Rdma_put(target_rank, _XMP_FJRDMA_TAG, raddr, laddr, rest, _XMP_COARRAY_FLAG_NIC);
      _XMP_add_num_of_puts();
    }
  }
}

/**
   The _XMP_FJMPI_Rdma_get() is a wrapper function of the FJMPI_Rdma_get().
   FJMPI_Rdma_get() cannot transfer more than _XMP_FJRDMA_MAX_SIZE(about 16MB) data.
   Thus, the _XMP_FJMPI_Rdma_get() executes mutiple put operations to trasfer more than 16MB data.
*/
static void _XMP_FJMPI_Rdma_get(const int target_rank, uint64_t raddr, uint64_t laddr,
				const size_t transfer_size)
{
  if(transfer_size <= _XMP_FJRDMA_MAX_SIZE){
    FJMPI_Rdma_get(target_rank, _XMP_FJRDMA_TAG, raddr, laddr, transfer_size, _XMP_COARRAY_FLAG_NIC);
    _XMP_add_num_of_gets();
  }
  else{
    int times = transfer_size / _XMP_FJRDMA_MAX_SIZE;
    int rest  = transfer_size - _XMP_FJRDMA_MAX_SIZE * times;

    for(int i=0;i<times;i++){
      FJMPI_Rdma_get(target_rank, _XMP_FJRDMA_TAG, raddr, laddr, _XMP_FJRDMA_MAX_SIZE, _XMP_COARRAY_FLAG_NIC);
      raddr += _XMP_FJRDMA_MAX_SIZE;
      laddr += _XMP_FJRDMA_MAX_SIZE;
      _XMP_add_num_of_gets();
    }

    if(rest != 0){
      FJMPI_Rdma_get(target_rank, _XMP_FJRDMA_TAG, raddr, laddr, rest, _XMP_COARRAY_FLAG_NIC);
      _XMP_add_num_of_gets();
    }
  }
}

#if defined(_FX10) || defined(_FX100)
/*********************************************************************************/
/* DESCRIPTION : Execute multiple put operation for FX10                         */
/* ARGUMENT    : [IN] target_rank    : Target rank                               */
/*               [IN] *raddrs        : Remote addresses                          */
/*               [IN] *laddrs        : Local addresses                           */
/*               [IN] *lengths       : Lengths                                   */
/*               [IN] stride         : Stride. If stride is 0, the first raadrs, */
/*                                     laddrs, and lengths is used               */
/*               [IN] transfer_elmts : Number of transfer elements               */
/* NOTE       : This function is used instead of FJMPI_Rdma_mput() which is used */
/*              on the K computer                                                */
/*********************************************************************************/
static void _FX10_Rdma_mput(const int target_rank, uint64_t *raddrs, uint64_t *laddrs,
                            const size_t *lengths, const int stride, const size_t transfer_elmts)
{
  if(stride == 0){
    for(size_t i=0;i<transfer_elmts;i++)
      _XMP_FJMPI_Rdma_put(target_rank, raddrs[i], laddrs[i], lengths[i]);
  }
  else{
    for(size_t i=0;i<transfer_elmts;i++){
      _XMP_FJMPI_Rdma_put(target_rank, raddrs[0], laddrs[0], lengths[0]);
      raddrs[0] += stride;
      laddrs[0] += stride;
    }
  }
}
#endif

/*********************************************************************************/
/* DESCRIPTION : Wrapper function for multiple put operations                    */
/* ARGUMENT    : [IN] target_rank    : Target rank                               */
/*               [IN] *raddrs        : Remote addresses                          */
/*               [IN] *laddrs        : Local addresses                           */
/*               [IN] *lengths       : Lengths                                   */
/*               [IN] stride         : Stride. If stride is 0, the first raadrs, */
/*                                     laddrs, and lengths is used               */
/*               [IN] transfer_elmts : Number of transfer elements               */
/*********************************************************************************/
static void _RDMA_mput(const size_t target_rank, uint64_t* raddrs, uint64_t* laddrs,
		       size_t* lengths, const int stride, const size_t transfer_elmts)
{
#if defined(_KCOMPUTER)
  FJMPI_Rdma_mput(target_rank, _XMP_FJRDMA_TAG, raddrs, laddrs, lengths, stride, transfer_elmts, _XMP_COARRAY_FLAG_NIC);
  _XMP_add_num_of_puts();
#elif defined(_FX10) || defined(_FX100)
  _FX10_Rdma_mput(target_rank, raddrs, laddrs, lengths, stride, transfer_elmts);
#endif
}

/************************************************************************/
/* DESCRIPTION : Execute scalar multiple put operation                  */
/* ARGUMENT    : [IN] target_rank    : Target rank                      */
/*               [IN] *raddrs        : Remote addresses                 */
/*               [IN] *laddrs        : Local addresses                  */
/*               [IN] *lengths       : Lengths                          */
/*               [IN] transfer_elmts : Number of transfer elements      */
/*               [IN] elmt_size      : Element size                     */
/************************************************************************/
static void _fjrdma_scalar_mput_do(const size_t target_rank, uint64_t* raddrs, uint64_t* laddrs,
                                   size_t* lengths, const size_t transfer_elmts, const size_t elmt_size)
{
  if(transfer_elmts <= _XMP_FJRDMA_MAX_MPUT){
    _RDMA_mput(target_rank, raddrs, laddrs, lengths, 0, transfer_elmts);
  }
  else{
    int times      = transfer_elmts / _XMP_FJRDMA_MAX_MPUT + 1;
    int rest_elmts = transfer_elmts - _XMP_FJRDMA_MAX_MPUT * (times - 1);
    for(int i=0;i<times;i++){
      size_t tmp_elmts = (i != times-1)? _XMP_FJRDMA_MAX_MPUT : rest_elmts;
      _RDMA_mput(target_rank, &raddrs[i*_XMP_FJRDMA_MAX_MPUT], &laddrs[i*_XMP_FJRDMA_MAX_MPUT],
		 &lengths[i*_XMP_FJRDMA_MAX_MPUT], 0, tmp_elmts);
    }
  }
}

/***********************************************************************/
/* DESCRIPTION : Execute malloc operation for coarray                  */
/* ARGUMENT    : [OUT] *coarray_desc  : Descriptor of new coarray      */
/*               [OUT] **addr         : Double pointer of new coarray  */
/*               [IN]  coarray_size   : Coarray size                   */
/***********************************************************************/
void _XMP_fjrdma_coarray_malloc(_XMP_coarray_t *coarray_desc, void **addr, const size_t coarray_size)
{
  *addr = _XMP_alloc(coarray_size);
  _XMP_fjrdma_regmem(coarray_desc, *addr, coarray_size);
}


/***********************************************************************/
/* DESCRIPTION : Register the local address of the coarray and get the */
/*               descriptor                                            */
/* ARGUMENT    : [OUT] *coarray_desc  : Descriptor of new coarray      */
/*               [IN]  *addr          : Pointer to the coarray         */
/*               [IN]  coarray_size   : Coarray size                   */
/***********************************************************************/
void _XMP_fjrdma_regmem(_XMP_coarray_t *coarray_desc, void *addr, const size_t coarray_size)
{
  uint64_t *each_addr = _XMP_alloc(sizeof(uint64_t) * _XMP_world_size);
  if(_memid == _XMP_FJRDMA_MAX_MEMID)
    _XMP_fatal("Too many coarrays. Number of coarrays is not more than 510.");

  coarray_desc->laddr = FJMPI_Rdma_reg_mem(_memid, addr, coarray_size);

  MPI_Barrier(MPI_COMM_WORLD);
  for(int ncount=0,i=1; i<_XMP_world_size+1; ncount++,i++){
    int partner_rank = (_XMP_world_rank + _XMP_world_size - i) % _XMP_world_size;
    each_addr[partner_rank] = FJMPI_Rdma_get_remote_addr(partner_rank, _memid);
    if(ncount > _XMP_INIT_RDMA_INTERVAL){
      MPI_Barrier(MPI_COMM_WORLD);
      ncount = 0;
    }
  }

  coarray_desc->real_addr = addr;
  coarray_desc->addr = (void *)each_addr;
  _memid++;
}

/**
   Deallocate memory region when calling _XMP_coarray_lastly_deallocate()
*/
void _XMP_fjrdma_coarray_lastly_deallocate()
{
  if(_memid == _XMP_FJRDMA_START_MEMID) return;

  _memid--;
  FJMPI_Rdma_dereg_mem(_memid);
}

/************************************************************************/
/* DESCRIPTION : Call put operation without preprocessing               */
/* ARGUMENT    : [IN] target_rank  : Target rank                        */
/*               [OUT] *dst_desc   : Descriptor of destination coarray  */
/*               [IN] *src_desc    : Descriptor of source coarray       */
/*               [IN] dst_offset   : Offset size of destination coarray */
/*               [IN] src_offset   : Offset size of source coarray      */
/*               [IN] dst_elmts    : Number of elements of destination  */
/*               [IN] src_elmts    : Number of elements of source       */
/*               [IN] elmt_size    : Element size                       */
/* NOTE       : Both dst and src are contiguous coarrays.               */
/*              target_rank != __XMP_world_rank.                        */
/* EXAMPLE    :                                                         */
/*     a[0:100]:[1] = b[0:100]; // a[] is a dst, b[] is a src           */
/************************************************************************/
void _XMP_fjrdma_contiguous_put(const int target_rank, const uint64_t dst_offset, const uint64_t src_offset,
				const _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc, 
				const size_t dst_elmts, const size_t src_elmts, const size_t elmt_size)
{
  size_t transfer_size = dst_elmts * elmt_size;
  _check_transfer_size(transfer_size);

  uint64_t raddr = (uint64_t)dst_desc->addr[target_rank] + dst_offset;
  uint64_t laddr = src_desc->laddr + src_offset;

  if(dst_elmts == src_elmts){
    _XMP_FJMPI_Rdma_put(target_rank, raddr, laddr, transfer_size);
  }
  else if(src_elmts == 1){
    uint64_t raddrs[dst_elmts], laddrs[dst_elmts];
    size_t lengths[dst_elmts];
    for(size_t i=0;i<dst_elmts;i++) raddrs[i]  = raddr + i * elmt_size;
    for(size_t i=0;i<dst_elmts;i++) laddrs[i]  = laddr;
    for(size_t i=0;i<dst_elmts;i++) lengths[i] = elmt_size;
    _fjrdma_scalar_mput_do(target_rank, raddrs, laddrs, lengths, dst_elmts, elmt_size);
  }
  else{
    _XMP_fatal("Coarray Error ! transfer size is wrong.\n");
  }

  if(_XMP_flag_put_nb == false)
    _XMP_fjrdma_sync_memory_put();
}

/*************************************************************************/
/* DESCRIPTION : Execute put operation without preprocessing             */
/* ARGUMENT    : [IN] target_rank   : Target rank                        */
/*               [IN] dst_offset    : Offset size of destination coarray */
/*               [IN] src_offset    : Offset size of source coarray      */
/*               [OUT] *dst_desc    : Descriptor of destination coarray  */
/*               [IN] *src_desc     : Descriptor of source coarray       */
/*               [IN] *src          : Pointer of source array            */
/*               [IN] transfer_size : Transfer size                      */
/* NOTE       : Both dst and src are contiguous arrays.                  */
/*              If src is NOT a coarray, src_desc is NULL.               */
/* EXAMPLE    :                                                          */
/*     a[0:100]:[1] = b[0:100]; // a[] is a dst, b[] is a src            */
/*************************************************************************/
static void _fjrdma_contiguous_put(const int target_rank, const uint64_t dst_offset, const uint64_t src_offset,
				   const _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc,
				   char *src, const size_t transfer_size)
{
  uint64_t raddr = (uint64_t)dst_desc->addr[target_rank] + dst_offset;
  uint64_t laddr;

  if(src_desc == NULL)
    laddr = FJMPI_Rdma_reg_mem(_XMP_TEMP_MEMID, src + src_offset, transfer_size);
  else
    laddr = src_desc->laddr + src_offset;

  _XMP_FJMPI_Rdma_put(target_rank, raddr, laddr, transfer_size);
  _XMP_fjrdma_sync_memory_put();

  if(src_desc == NULL)   
    FJMPI_Rdma_dereg_mem(_XMP_TEMP_MEMID);
}

/*********************************************************************************/
/* DESCRIPTION : Execute scalar multiple put operation                           */
/* ARGUMENT    : [IN] target_rank    : Target rank                               */
/*               [IN] dst_offset     : Offset size of destination array          */
/*               [IN] src_offset     : Offset size of source array               */
/*               [IN] *dst_info      : Information of destination array          */
/*               [IN] dst_dims       : Number of dimensions of destination array */
/*               [OUT] *dst_desc     : Descriptor of destination coarray         */
/*               [IN] *src_desc      : Descriptor of source coarray              */
/*               [IN] *src           : Pointer of source array                   */
/*               [IN] transfer_elmts : Number of transfer elements               */
/* NOTE       : If src is NOT a coarray, src_desc is NULL.                       */
/* EXAMPLE    :                                                                  */
/*     a[0:100:2]:[1] = b[0]; // a[] is a dst, b[] is a src                      */
/*********************************************************************************/
static void _fjrdma_scalar_mput(const int target_rank, const uint64_t dst_offset, const uint64_t src_offset, 
				const _XMP_array_section_t *dst_info, const int dst_dims,
				const _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc, 
				char *src, const size_t transfer_elmts)
{
  uint64_t raddr = (uint64_t)dst_desc->addr[target_rank] + dst_offset;
  uint64_t laddr;
  size_t elmt_size = dst_desc->elmt_size;
  uint64_t raddrs[transfer_elmts], laddrs[transfer_elmts];
  size_t lengths[transfer_elmts];

  if(src_desc == NULL)
    laddr = FJMPI_Rdma_reg_mem(_XMP_TEMP_MEMID, src + src_offset, elmt_size);
  else
    laddr = src_desc->laddr + src_offset;

  // Set parameters for FJMPI_Rdma_mput
  _XMP_set_coarray_addresses(raddr, dst_info, dst_dims, transfer_elmts, raddrs);
  for(size_t i=0;i<transfer_elmts;i++) laddrs[i] = laddr;
  for(size_t i=0;i<transfer_elmts;i++) lengths[i] = elmt_size;

  _fjrdma_scalar_mput_do(target_rank, raddrs, laddrs, lengths, transfer_elmts, elmt_size);
  _XMP_fjrdma_sync_memory_put();

  if(src_desc == NULL)
    FJMPI_Rdma_dereg_mem(_XMP_TEMP_MEMID);
}

/*******************************************************************/
/* DESCRIPTION : Get array size                                   */
/* ARGUMENT    : [IN] *array_info : Information of array          */
/*               [IN] dims        : Number of dimensions of array */
/* RETURN     : Array size                                        */
/* EXAMPLE    :                                                   */
/*     int a[10][20]; -> 800                                      */
/******************************************************************/
static size_t _get_array_size(const _XMP_array_section_t *array_info, const int dims)
{
  return array_info[0].distance * array_info[0].elmts;
}

/*********************************************************************/
/* DESCRIPTION : Execute multiple put operation with the same stride */
/* ARGUMENT    : [IN] target_rank : Target rank                      */
/*               [IN] raddr       : Remote address                   */
/*               [IN] laddr       : Local address                    */
/*               [IN] *array_info : Information of array             */
/*               [IN] *array_dims : Number of dimensions             */
/*               [IN] elmt_size   : Element size                     */
/* NOTE       : The sixth argument of FJMPI_Rdma_mput() can NOT be 0 */
/* EXAMPLE    :                                                      */
/*     a[0:10:2]:[2] = b[2:10:2]; // a[] is a dst, b[] is a src      */
/*********************************************************************/
static void _fjrdma_NON_contiguous_the_same_stride_mput(const int target_rank, uint64_t raddr, uint64_t laddr,
							const size_t transfer_elmts, const _XMP_array_section_t *array_info,
							const int array_dims, size_t elmt_size)
{
  size_t copy_chunk_dim = (size_t)_XMP_get_dim_of_allelmts(array_dims, array_info);
  size_t copy_chunk     = (size_t)_XMP_calc_copy_chunk(copy_chunk_dim, array_info);
  size_t copy_elmts     = transfer_elmts/(copy_chunk/elmt_size);
  size_t stride         = _XMP_calc_stride(array_info, array_dims, copy_chunk);

  if(copy_elmts <= _XMP_FJRDMA_MAX_MPUT){
    _RDMA_mput(target_rank, &raddr, &laddr, &copy_chunk, stride, copy_elmts);
  }
  else{
    int times      = copy_elmts / _XMP_FJRDMA_MAX_MPUT + 1;
    int rest_elmts = copy_elmts - _XMP_FJRDMA_MAX_MPUT * (times - 1);
    size_t tmp_elmts;

    for(int i=0;i<times;i++){
      uint64_t tmp_raddr = raddr + (i*_XMP_FJRDMA_MAX_MPUT*stride);
      uint64_t tmp_laddr = laddr + (i*_XMP_FJRDMA_MAX_MPUT*stride);
      tmp_elmts = (i != times-1)? _XMP_FJRDMA_MAX_MPUT : rest_elmts;
      _RDMA_mput(target_rank, &tmp_raddr, &tmp_laddr, &copy_chunk, stride, tmp_elmts);
    }
  }
}

/*********************************************************************************/
/* DESCRIPTION : Execute multiple put operation in general                       */
/* ARGUMENT    : [IN] target_rank    : Target rank                               */
/*               [IN] raddr          : Remote address                            */
/*               [IN] laddr          : Local address                             */
/*               [IN] transfer_elmts : Number of transfer elements               */
/*               [IN] *dst_info      : Information of destination array          */
/*               [IN] *src_info      : Information of source array               */
/*               [IN] dst_dims       : Number of dimensions of destination array */
/*               [IN] src_dims       : Number of dimensions of source array      */
/*               [IN] elmt_size      : Element size                              */
/*********************************************************************************/
static void _fjrdma_NON_contiguous_general_mput(const int target_rank, uint64_t raddr, uint64_t laddr,
						const size_t transfer_elmts,
						const _XMP_array_section_t *dst_info, const _XMP_array_section_t *src_info,
						const int dst_dims, const int src_dims, size_t elmt_size)
{
  size_t copy_chunk = _XMP_calc_max_copy_chunk(dst_dims, src_dims, dst_info, src_info);
  size_t copy_elmts = transfer_elmts/(copy_chunk/elmt_size);
  uint64_t raddrs[copy_elmts], laddrs[copy_elmts];
  size_t   lengths[copy_elmts];

  // Set parameters for FJMPI_Rdma_mput
  _XMP_set_coarray_addresses_with_chunk(raddrs, raddr, dst_info, dst_dims, copy_chunk, copy_elmts);
  _XMP_set_coarray_addresses_with_chunk(laddrs, laddr, src_info, src_dims, copy_chunk, copy_elmts);
  for(size_t i=0;i<copy_elmts;i++) lengths[i] = copy_chunk;

  _fjrdma_scalar_mput_do(target_rank, raddrs, laddrs, lengths, copy_elmts, elmt_size);
}

/*********************************************************************************/
/* DESCRIPTION : Execute put operation for NON-contiguous region                 */
/* ARGUMENT    : [IN] target_rank    : Target rank                               */
/*               [IN] dst_offset     : Offset size of destination array          */
/*               [IN] src_offset     : Offset size of source array               */
/*               [IN] *dst_info      : Information of destination array          */
/*               [IN] *src_info      : Information of source array               */
/*               [IN] dst_dims       : Number of dimensions of destination array */
/*               [IN] src_dims       : Number of dimensions of source array      */
/*               [OUT] *dst_desc     : Descriptor of destination array           */
/*               [IN] *src_desc      : Descriptor of source array                */
/*               [IN] *src           : Pointer of source array                   */
/*               [IN] transfer_elmts : Number of transfer elements               */
/* NOTE       : src and/or dst arrays are NOT contiguous.                        */
/*              If src is NOT a coarray, src_desc is NULL                        */
/*********************************************************************************/
static void _fjrdma_NON_contiguous_put(const int target_rank, const uint64_t dst_offset, const uint64_t src_offset,
				       const _XMP_array_section_t *dst_info, const _XMP_array_section_t *src_info,
				       const int dst_dims, const int src_dims, 
				       const _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc, 
				       void *src, const size_t transfer_elmts)
{
  uint64_t raddr = (uint64_t)dst_desc->addr[target_rank] + dst_offset;
  uint64_t laddr;
  size_t elmt_size = dst_desc->elmt_size;

  if(src_desc == NULL){
    size_t array_size = _get_array_size(src_info, src_dims);
    laddr = FJMPI_Rdma_reg_mem(_XMP_TEMP_MEMID, src, array_size) + src_offset;
  }
  else{
    laddr = src_desc->laddr + src_offset;
  }

  if(_XMP_is_the_same_constant_stride(dst_info, src_info, dst_dims, src_dims)){
    _fjrdma_NON_contiguous_the_same_stride_mput(target_rank, raddr, laddr, transfer_elmts,
						dst_info, dst_dims, elmt_size);
  }
  else{
    _fjrdma_NON_contiguous_general_mput(target_rank, raddr, laddr, transfer_elmts, 
    					dst_info, src_info, dst_dims, src_dims, elmt_size);
  }
  
  _XMP_fjrdma_sync_memory_put();

  if(src_desc == NULL)
    FJMPI_Rdma_dereg_mem(_XMP_TEMP_MEMID);
}

/***************************************************************************************/
/* DESCRIPTION : Execute put operation                                                 */
/* ARGUMENT    : [IN] dst_contiguous : Is destination region contiguous ? (TRUE/FALSE) */
/*               [IN] src_contiguous : Is source region contiguous ? (TRUE/FALSE)      */
/*               [IN] target_rank    : Target rank                                     */
/*               [IN] dst_dims       : Number of dimensions of destination array       */
/*               [IN] src_dims       : Number of dimensions of source array            */
/*               [IN] *dst_info      : Information of destination array                */
/*               [IN] *src_info      : Information of source array                     */
/*               [OUT] *dst_desc     : Descriptor of destination coarray               */
/*               [IN] *src_desc      : Descriptor of source array                      */
/*               [IN] *src           : Pointer of source array                         */
/*               [IN] dst_elmts      : Number of elements of destination array         */
/*               [IN] src_elmts      : Number of elements of source array              */
/***************************************************************************************/
void _XMP_fjrdma_put(const int dst_contiguous, const int src_contiguous, const int target_rank, 
		     const int dst_dims, const int src_dims, const _XMP_array_section_t *dst_info, 
		     const _XMP_array_section_t *src_info, const _XMP_coarray_t *dst_desc, 
		     const _XMP_coarray_t *src_desc, void *src, const size_t dst_elmts, const size_t src_elmts)
{
  uint64_t dst_offset = (uint64_t)_XMP_get_offset(dst_info, dst_dims);
  uint64_t src_offset = (uint64_t)_XMP_get_offset(src_info, src_dims);
  size_t transfer_size = dst_desc->elmt_size * dst_elmts;
  _check_transfer_size(transfer_size);

  if(dst_elmts == src_elmts){
    if(dst_contiguous == _XMP_N_INT_TRUE && src_contiguous == _XMP_N_INT_TRUE){
      _fjrdma_contiguous_put(target_rank, dst_offset, src_offset, dst_desc, src_desc, src, transfer_size);
    }
    else{
      _fjrdma_NON_contiguous_put(target_rank, dst_offset, src_offset, dst_info, src_info, dst_dims, src_dims, 
      				 dst_desc, src_desc, src, dst_elmts);
    }
  }
  else{
    if(src_elmts == 1){
      _fjrdma_scalar_mput(target_rank, dst_offset, src_offset, dst_info, dst_dims, dst_desc, src_desc, 
			  src, dst_elmts);
    }
    else{
      _XMP_fatal("Number of elements is invalid");
    }
  }
}

/************************************************************************/
/* DESCRIPTION : Execute get operation without preprocessing            */
/* ARGUMENT    : [IN] target_rank  : Target rank                        */
/*               [OUT] *dst_desc   : Descriptor of destination coarray  */
/*               [IN] *src_desc    : Descriptor of source coarray       */
/*               [IN] dst_offset   : Offset size of destination coarray */
/*               [IN] src_offset   : Offset size of source coarray      */
/*               [IN] dst_elmts    : Number of elements of destination  */
/*               [IN] src_elmts    : Number of elements of source       */
/*               [IN] elmt_size    : Element size                       */
/* NOTE       : Both dst and src are contiguous coarrays.               */
/*              target_rank != __XMP_world_rank.                        */
/* EXAMPLE    :                                                         */
/*     a[0:100] = b[0:100]:[1]; // a[] is a dst, b[] is a src           */
/************************************************************************/
void _XMP_fjrdma_contiguous_get(const int target_rank, const _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc,
				const uint64_t dst_offset, const uint64_t src_offset,
				const size_t dst_elmts, const size_t src_elmts, const size_t elmt_size)
{
  size_t transfer_size = dst_elmts * elmt_size;
  _check_transfer_size(transfer_size);

  uint64_t raddr = (uint64_t)src_desc->addr[target_rank] + src_offset;
  uint64_t laddr = dst_desc->laddr + dst_offset;
  
  if(dst_elmts == src_elmts){
    _XMP_FJMPI_Rdma_get(target_rank, raddr, laddr, transfer_size);
    _XMP_fjrdma_sync_memory_get();
  }
  else if(src_elmts == 1){
    _XMP_FJMPI_Rdma_get(target_rank, raddr, laddr, elmt_size);
    _XMP_fjrdma_sync_memory_get();

    char *dst = dst_desc->real_addr + dst_offset;
    for(size_t i=1;i<dst_elmts;i++)
      memcpy(dst+i*elmt_size, dst, elmt_size);
  }
  else{
    _XMP_fatal("Coarray Error ! transfer size is wrong.\n");
  }
}

/************************************************************************/
/* DESCRIPTION : Execute get operation for contiguous region            */
/* ARGUMENT    : [IN] target_rank   : Target rank                       */
/*               [IN] dst_offset    : Offset size of destination array  */
/*               [IN] src_offset    : Offset size of source array       */
/*               [OUT] *dst         : Pointer of destination array      */
/*               [IN] *dst_desc     : Descriptor of destination coarray */
/*               [IN] *src_desc     : Descriptor of source coarray      */
/*               [IN] transfer_size : Transfer size                     */
/* NOTE       : Both dst and src are contiguous arrays.                 */
/*              If dst is NOT a coarray, dst_desc is NULL.              */
/* EXAMPLE    :                                                         */
/*     a[0:100] = b[0:100]:[1]; // a[] is a dst, b[] is a src           */
/************************************************************************/
static void _fjrdma_contiguous_get(const int target_rank, const uint64_t dst_offset, const uint64_t src_offset,
				   char *dst, const _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc, 
				   const size_t transfer_size)
{
  uint64_t raddr = (uint64_t)src_desc->addr[target_rank] + src_offset;
  uint64_t laddr;

  if(dst_desc == NULL){
    laddr = FJMPI_Rdma_reg_mem(_XMP_TEMP_MEMID, dst + dst_offset, transfer_size);
  }
  else
    laddr = dst_desc->laddr + dst_offset;
  
  _XMP_FJMPI_Rdma_get(target_rank, raddr, laddr, transfer_size);
  _XMP_fjrdma_sync_memory_get();

  if(dst_desc == NULL)
    FJMPI_Rdma_dereg_mem(_XMP_TEMP_MEMID);
}

/*********************************************************************************/
/* DESCRIPTION : Execute get operation for NON-contiguous region                 */
/* ARGUMENT    : [IN] target_rank    : Target rank                               */
/*               [IN] dst_offset     : Offset size of destination array          */
/*               [IN] src_offset     : Offset size of source array               */
/*               [IN] dst_info       : Information of destination array          */
/*               [IN] src_info       : Information of source array               */
/*               [OUT] *dst          : Pointer of destination array              */
/*               [IN] *dst_desc      : Descriptor of destination coarray         */
/*               [IN] *src_desc      : Descriptor of source coarray              */
/*               [IN] dst_dims       : Number of dimensions of destination array */
/*               [IN] src_dims       : Number of dimensions of source array      */
/*               [IN] transfer_elmts : Number of transfer elements               */
/* NOTE       : If dst is NOT a coarray, dst_desc is NULL.                       */
/* EXAMPLE    :                                                                  */
/*     a[0:100:2] = b[0:100:2]:[1]; // a[] is a dst, b[] is a src                */
/*********************************************************************************/
static void _fjrdma_NON_contiguous_get(const int target_rank, const uint64_t dst_offset, const uint64_t src_offset,
				       const _XMP_array_section_t *dst_info, const _XMP_array_section_t *src_info,
				       void *dst, const _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc,
				       const int dst_dims, const int src_dims, const int transfer_elmts)
{
  size_t copy_chunk = _XMP_calc_max_copy_chunk(dst_dims, src_dims, dst_info, src_info);
  size_t elmt_size  = src_desc->elmt_size;
  size_t copy_elmts = transfer_elmts/(copy_chunk/elmt_size);
  uint64_t raddr = (uint64_t)src_desc->addr[target_rank] + src_offset;
  uint64_t laddr;
  uint64_t raddrs[copy_elmts], laddrs[copy_elmts];

  if(dst_desc == NULL){
    size_t array_size = _get_array_size(dst_info, dst_dims);
    laddr = FJMPI_Rdma_reg_mem(_XMP_TEMP_MEMID, dst, array_size) + dst_offset;
  }
  else{
    laddr = dst_desc->laddr + dst_offset;
  }

  // Set parameters for multipul FJMPI_Rdma_get()
  _XMP_set_coarray_addresses_with_chunk(raddrs, raddr, src_info, src_dims, copy_chunk, copy_elmts);
  _XMP_set_coarray_addresses_with_chunk(laddrs, laddr, dst_info, dst_dims, copy_chunk, copy_elmts);

  if(copy_elmts <= _XMP_FJRDMA_MAX_MGET){
    for(size_t i=0;i<copy_elmts;i++)
      _XMP_FJMPI_Rdma_get(target_rank, raddrs[i], laddrs[i], copy_chunk);
  }
  else{
    int times      = copy_elmts / _XMP_FJRDMA_MAX_MGET + 1;
    int rest_elmts = copy_elmts - _XMP_FJRDMA_MAX_MGET * (times - 1);

    for(int i=0;i<times;i++){
      size_t tmp_elmts = (i != times-1)? _XMP_FJRDMA_MAX_MGET : rest_elmts;
      for(int j=0;j<tmp_elmts;j++)
	_XMP_FJMPI_Rdma_get(target_rank, raddrs[j+i*_XMP_FJRDMA_MAX_MGET], laddrs[j+i*_XMP_FJRDMA_MAX_MGET], copy_chunk);
    }
  }
  _XMP_fjrdma_sync_memory_get();

  if(dst_desc == NULL)
    FJMPI_Rdma_dereg_mem(_XMP_TEMP_MEMID);
}

/*********************************************************************************/
/* DESCRIPTION : Execute scalar multiple get operation                           */
/* ARGUMENT    : [IN] target_rank    : Target rank                               */
/*               [IN] dst_offset     : Offset size of destination array          */
/*               [IN] src_offset     : Offset size of source array               */
/*               [IN] dst_info       : Information of destination array          */
/*               [IN] dst_dims       : Number of dimensions of destination array */
/*               [OUT] *dst_desc     : Descriptor of destination array           */
/*               [IN] *src_desc      : Descriptor of source array                */
/*               [IN] *dst           : Pointer of destination array              */
/*               [IN] transfer_elmts : Number of transfer elements               */
/* NOTE       : If dst is NOT a coarray, dst_desc != NULL                        */
/* EXAMPLE    :                                                                  */
/*     a[0:100]:[1] = b[0]; // a[] is a dst, b[] is a src                        */
/*********************************************************************************/
static void _fjrdma_scalar_mget(const int target_rank, const uint64_t dst_offset, const uint64_t src_offset,
				const _XMP_array_section_t *dst_info, const int dst_dims,
				const _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc,
				char *dst, const size_t transfer_elmts)
{
  uint64_t raddr = (uint64_t)src_desc->addr[target_rank] + src_offset;
  uint64_t laddr;
  size_t elmt_size = src_desc->elmt_size;

  if(dst_desc == NULL)
    laddr = FJMPI_Rdma_reg_mem(_XMP_TEMP_MEMID, dst + dst_offset, elmt_size);
  else
    laddr = dst_desc->laddr + dst_offset;

  _XMP_FJMPI_Rdma_get(target_rank, raddr, laddr, elmt_size);
  _XMP_fjrdma_sync_memory_get();

  // Local copy (Note that number of copies is one time more in following _XMP_stride_memcpy_Xdim())
  char *src_addr = dst + dst_offset;
  char *dst_addr = src_addr;
  switch (dst_dims){
  case 1:
    _XMP_stride_memcpy_1dim(dst_addr, src_addr, dst_info, elmt_size, _XMP_SCALAR_MCOPY);
    break;
  case 2:
    _XMP_stride_memcpy_2dim(dst_addr, src_addr, dst_info, elmt_size, _XMP_SCALAR_MCOPY);
    break;
  case 3:
    _XMP_stride_memcpy_3dim(dst_addr, src_addr, dst_info, elmt_size, _XMP_SCALAR_MCOPY);
    break;
  case 4:
    _XMP_stride_memcpy_4dim(dst_addr, src_addr, dst_info, elmt_size, _XMP_SCALAR_MCOPY);
    break;
  case 5:
    _XMP_stride_memcpy_5dim(dst_addr, src_addr, dst_info, elmt_size, _XMP_SCALAR_MCOPY);
    break;
  case 6:
    _XMP_stride_memcpy_6dim(dst_addr, src_addr, dst_info, elmt_size, _XMP_SCALAR_MCOPY);
    break;
  case 7:
    _XMP_stride_memcpy_7dim(dst_addr, src_addr, dst_info, elmt_size, _XMP_SCALAR_MCOPY);
    break;
  default:
    _XMP_fatal("Coarray Error ! Dimension is too big.\n");
    break;
  }

  if(dst_desc == NULL)
    FJMPI_Rdma_dereg_mem(_XMP_TEMP_MEMID);
}

/***************************************************************************************/
/* DESCRIPTION : Execute put operation                                                 */
/* ARGUMENT    : [IN] src_contiguous : Is source region contiguous ? (TRUE/FALSE)      */
/*               [IN] dst_contiguous : Is destination region contiguous ? (TRUE/FALSE) */
/*               [IN] target_rank    : Target rank                                     */
/*               [IN] src_dims       : Number of dimensions of source array            */
/*               [IN] dst_dims       : Number of dimensions of destination array       */
/*               [IN] *src_info      : Information of source array                     */
/*               [IN] *dst_info      : Information of destination array                */
/*               [IN] *src_desc      : Descriptor of source array                      */
/*               [OUT] *dst_desc     : Descriptor of destination coarray               */
/*               [IN] *dst           : Pointer of destination array                    */
/*               [IN] src_elmts      : Number of elements of source array              */
/*               [IN] dst_elmts      : Number of elements of destination array         */
/***************************************************************************************/
void _XMP_fjrdma_get(const int src_contiguous, const int dst_contiguous, const int target_rank, 
		     const int src_dims, const int dst_dims, 
		     const _XMP_array_section_t *src_info, const _XMP_array_section_t *dst_info, 
		     const _XMP_coarray_t *src_desc, const _XMP_coarray_t *dst_desc, void *dst,
		     const size_t src_elmts, const size_t dst_elmts)
{
  uint64_t dst_offset = (uint64_t)_XMP_get_offset(dst_info, dst_dims);
  uint64_t src_offset = (uint64_t)_XMP_get_offset(src_info, src_dims);
  size_t transfer_size = src_desc->elmt_size * src_elmts;

  _check_transfer_size(transfer_size);

  if(src_elmts == dst_elmts){
    if(dst_contiguous == _XMP_N_INT_TRUE && src_contiguous == _XMP_N_INT_TRUE){
      _fjrdma_contiguous_get(target_rank, dst_offset, src_offset, dst, dst_desc, src_desc, transfer_size);
    }
    else{
      _fjrdma_NON_contiguous_get(target_rank, dst_offset, src_offset, dst_info, src_info,
				 dst, dst_desc, src_desc, dst_dims, src_dims, src_elmts);
    }
  }
  else{
    if(src_elmts == 1){
      _fjrdma_scalar_mget(target_rank, dst_offset, src_offset, dst_info, dst_dims, dst_desc, src_desc, (char *)dst, dst_elmts);
    }
    else{
      _XMP_fatal("Number of elements is invalid");
    }
  }
}

/**
 * Build table and Initialize for sync images
 */
void _XMP_fjrdma_build_sync_images_table()
{
  _sync_images_table = malloc(sizeof(unsigned int) * _XMP_world_size);

  for(int i=0;i<_XMP_world_size;i++)
    _sync_images_table[i] = 0;

  double *token     = _XMP_alloc(sizeof(double));
  _local_rdma_addr  = FJMPI_Rdma_reg_mem(_XMP_SYNC_IMAGES_ID, token, sizeof(double));
  _remote_rdma_addr = _XMP_alloc(sizeof(uint64_t) * _XMP_world_size);

  // Obtain remote RDMA addresses
  MPI_Barrier(MPI_COMM_WORLD);
  for(int ncount=0,i=1; i<_XMP_world_size+1; ncount++,i++){
    int partner_rank = (_XMP_world_rank + _XMP_world_size - i) % _XMP_world_size;
    _remote_rdma_addr[partner_rank] = FJMPI_Rdma_get_remote_addr(partner_rank, _XMP_SYNC_IMAGES_ID);

    if(ncount > _XMP_INIT_RDMA_INTERVAL){
      MPI_Barrier(MPI_COMM_WORLD);
      ncount = 0;
    }
  }
}

/**
   Add rank to table
   *
   * @param[in]  rank rank number
   */
static void _add_sync_images_table(const int rank)
{
  _sync_images_table[rank]++;
}

/**
   Notify to nodes
   *
   * @param[in]  num        number of nodes
   * @param[in]  *rank_set  rank set
   */
static void _notify_sync_images(const int num, int *rank_set)
{
  int num_of_requests = 0;
  
  for(int i=0;i<num;i++)
    if(rank_set[i] == _XMP_world_rank){
      _add_sync_images_table(_XMP_world_rank);
    }
    else{
      FJMPI_Rdma_put(rank_set[i], _XMP_SYNC_IMAGES_TAG, _remote_rdma_addr[rank_set[i]], 
		     _local_rdma_addr, sizeof(double), _XMP_SYNC_IMAGES_FLAG_NIC);
      num_of_requests++;
    }

  for(int i=0;i<num_of_requests;i++)
    while(FJMPI_Rdma_poll_cq(_XMP_SYNC_IMAGES_SEND_NIC, NULL) != FJMPI_RDMA_NOTICE);  // Wait until finishing above put operations
}

/**
   Check to recieve all request from all node
   *
   * @param[in]  num        number of nodes
   * @param[in]  *rank_set  rank set
   * @praam[in]  old_wait_sync_images[num] old images set
*/
static _Bool _check_sync_images_table(const int num, int *rank_set)
{
  int checked = 0;

  for(int i=0;i<num;i++)
    if(_sync_images_table[rank_set[i]] > 0)
      checked++;

  if(checked == num) return true;
  else               return false;
}

/**
   Wait until recieving all request from all node
   *
   * @param[in]  num                       number of nodes
   * @param[in]  *rank_set                 rank set
   * @praam[in]  old_wait_sync_images[num] old images set
*/
static void _wait_sync_images(const int num, int *rank_set)
{
  struct FJMPI_Rdma_cq cq;

  while(1){
    if(_check_sync_images_table(num, rank_set)) break;
    
    if(FJMPI_Rdma_poll_cq(_XMP_SYNC_IMAGES_RECV_NIC, &cq) == FJMPI_RDMA_HALFWAY_NOTICE)
      _add_sync_images_table(cq.pid);
  }
}

/**
   Execute sync images
   *
   * @param[in]  num         number of nodes
   * @param[in]  *image_set  image set
   * @param[out] status      status
*/
void _XMP_fjrdma_sync_images(const int num, int* image_set, int* status)
{
  _XMP_fjrdma_sync_memory();

  if(num == 0){
    return;
  }
  else if(num < 0){
    fprintf(stderr, "Invalid value is used in xmp_sync_memory. The first argument is %d\n", num);
    _XMP_fatal_nomsg();
  }

  _notify_sync_images(num, image_set);
  _wait_sync_images(num, image_set);

  // Update table for post-processing
  for(int i=0;i<num;i++)
    _sync_images_table[image_set[i]]--;
}
