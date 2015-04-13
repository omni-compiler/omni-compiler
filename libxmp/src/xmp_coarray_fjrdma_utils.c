#include <stdlib.h>
#include "xmp_internal.h"

uint64_t* _XMP_calc_raddrs_mput(const uint64_t raddr, const _XMP_array_section_t *dst_info, const int dst_dims, 
				const size_t elmts)
{
  uint64_t* raddrs = malloc(sizeof(uint64_t) * elmts);
  uint64_t tmp_start_offset[dst_dims], tmp_stride_offset[dst_dims], tmp_offset[dst_dims];

  // Temporally variables to reduce calculation for offset
  for(int i=0;i<dst_dims;i++){
    tmp_start_offset[i]  = dst_info[i].start  * dst_info[i].distance;
    tmp_stride_offset[i] = dst_info[i].stride * dst_info[i].distance;
  }
 
  int num = 0;
  switch (dst_dims){
  case 1:
    for(int i=0;i<dst_info[0].length;i++){
      tmp_offset[0] = tmp_start_offset[0] + i * tmp_stride_offset[0];
      raddrs[num++] = raddr + tmp_offset[0];
    }
    break;;
  case 2:
    for(int i=0;i<dst_info[0].length;i++){
      tmp_offset[0] = tmp_start_offset[0] + i * tmp_stride_offset[0];
      for(int j=0;j<dst_info[1].length;j++){
        tmp_offset[1] = tmp_start_offset[1] + j * tmp_stride_offset[1] + tmp_offset[0];
	raddrs[num++] = raddr + tmp_offset[1];
      }
    }
    break;;
  case 3:
    for(int i=0;i<dst_info[0].length;i++){
      tmp_offset[0] = tmp_start_offset[0] + i * tmp_stride_offset[0];
      for(int j=0;j<dst_info[1].length;j++){
        tmp_offset[1] = tmp_start_offset[1] + j * tmp_stride_offset[1] + tmp_offset[0];
        for(int k=0;k<dst_info[2].length;k++){
          tmp_offset[2] = tmp_start_offset[2] + k * tmp_stride_offset[2] + tmp_offset[1];
	  raddrs[num++] = raddr + tmp_offset[2];
	}
      }
    }
    break;;
  case 4:
    for(int i=0;i<dst_info[0].length;i++){
      tmp_offset[0] = tmp_start_offset[0] + i * tmp_stride_offset[0];
      for(int j=0;j<dst_info[1].length;j++){
        tmp_offset[1] = tmp_start_offset[1] + j * tmp_stride_offset[1] + tmp_offset[0];
        for(int k=0;k<dst_info[2].length;k++){
          tmp_offset[2] = tmp_start_offset[2] + k * tmp_stride_offset[2] + tmp_offset[1];
          for(int l=0;l<dst_info[3].length;l++){
            tmp_offset[3] = tmp_start_offset[3] + l * tmp_stride_offset[3] + tmp_offset[2];
	    raddrs[num++] = raddr + tmp_offset[3];
	  }
        }
      }
    }
    break;;
  case 5:
    for(int i=0;i<dst_info[0].length;i++){
      tmp_offset[0] = tmp_start_offset[0] + i * tmp_stride_offset[0];
      for(int j=0;j<dst_info[1].length;j++){
        tmp_offset[1] = tmp_start_offset[1] + j * tmp_stride_offset[1] + tmp_offset[0];
        for(int k=0;k<dst_info[2].length;k++){
          tmp_offset[2] = tmp_start_offset[2] + k * tmp_stride_offset[2] + tmp_offset[1];
          for(int l=0;l<dst_info[3].length;l++){
            tmp_offset[3] = tmp_start_offset[3] + l * tmp_stride_offset[3] + tmp_offset[2];
            for(int m=0;m<dst_info[4].length;m++){
              tmp_offset[4] = tmp_start_offset[4] + m * tmp_stride_offset[4] + tmp_offset[3];
	      raddrs[num++] = raddr + tmp_offset[4];
	    }
          }
        }
      }
    }
    break;;
  case 6:
    for(int i=0;i<dst_info[0].length;i++){
      tmp_offset[0] = tmp_start_offset[0] + i * tmp_stride_offset[0];
      for(int j=0;j<dst_info[1].length;j++){
        tmp_offset[1] = tmp_start_offset[1] + j * tmp_stride_offset[1] + tmp_offset[0];
        for(int k=0;k<dst_info[2].length;k++){
          tmp_offset[2] = tmp_start_offset[2] + k * tmp_stride_offset[2] + tmp_offset[1];
          for(int l=0;l<dst_info[3].length;l++){
            tmp_offset[3] = tmp_start_offset[3] + l * tmp_stride_offset[3] + tmp_offset[2];
            for(int m=0;m<dst_info[4].length;m++){
              tmp_offset[4] = tmp_start_offset[4] + m * tmp_stride_offset[4] + tmp_offset[3];
              for(int n=0;n<dst_info[5].length;n++){
                tmp_offset[5] = tmp_start_offset[5] + n * tmp_stride_offset[5] + tmp_offset[4];
		raddrs[num++] = raddr + tmp_offset[5];
	      }
            }
          }
        }
      }
    }
    break;;
  case 7:
    for(int i=0;i<dst_info[0].length;i++){
      tmp_offset[0] = tmp_start_offset[0] + i * tmp_stride_offset[0];
      for(int j=0;j<dst_info[1].length;j++){
        tmp_offset[1] = tmp_start_offset[1] + j * tmp_stride_offset[1] + tmp_offset[0];
        for(int k=0;k<dst_info[2].length;k++){
          tmp_offset[2] = tmp_start_offset[2] + k * tmp_stride_offset[2] + tmp_offset[1];
          for(int l=0;l<dst_info[3].length;l++){
            tmp_offset[3] = tmp_start_offset[3] + l * tmp_stride_offset[3] + tmp_offset[2];
            for(int m=0;m<dst_info[4].length;m++){
              tmp_offset[4] = tmp_start_offset[4] + m * tmp_stride_offset[4] + tmp_offset[3];
              for(int n=0;n<dst_info[5].length;n++){
                tmp_offset[5] = tmp_start_offset[5] + n * tmp_stride_offset[5] + tmp_offset[4];
                for(int p=0;p<dst_info[6].length;p++){
                  tmp_offset[6] = tmp_start_offset[6] + p * tmp_stride_offset[6] + tmp_offset[5];
		  raddrs[num++] = raddr + tmp_offset[6];
		}
              }
            }
          }
        }
      }
    }
    break;;
  }

  return raddrs;
}

uint64_t* _XMP_calc_laddrs_mput(const uint64_t laddr, const size_t elmts)
{
  uint64_t *laddrs = malloc(sizeof(uint64_t) * elmts);

  for(int i=0;i<elmts;i++)
    laddrs[i] = laddr;

  return laddrs;
}

size_t* _XMP_calc_lengths_mput(const size_t length, const size_t elmts)
{
  size_t *lengths = malloc(sizeof(size_t) * elmts);

  for(int i=0;i<elmts;i++)
    lengths[i] = length;
   
  return lengths;
}
