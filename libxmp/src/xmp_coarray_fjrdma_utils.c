#include <stdlib.h>
#include "xmp_internal.h"

void _XMP_set_coarray_addresses(const uint64_t addr, const _XMP_array_section_t *array, const int dims, 
				const size_t elmts, uint64_t* addrs)
{
  uint64_t stride_offset[dims], tmp[dims];

  // Temporally variables to reduce calculation for offset
  for(int i=0;i<dims;i++)
    stride_offset[i] = array[i].stride * array[i].distance;
 
  int num = 0;
  switch (dims){
  case 1:
    for(int i=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      addrs[num++] = addr + tmp[0];
    }
    break;;
  case 2:
    for(int i=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
	addrs[num++] = addr + tmp[0] + tmp[1];
      }
    }
    break;;
  case 3:
    for(int i=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
	for(int k=0;k<array[2].length;k++){
	  tmp[2] = stride_offset[2] * k;
	  addrs[num++] = addr + tmp[0] + tmp[1] + tmp[2];
	}
      }
    }
    break;;
  case 4:
    for(int i=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
	for(int k=0;k<array[2].length;k++){
          tmp[2] = stride_offset[2] * k;
	  for(int l=0;l<array[3].length;l++){
	    tmp[3] = stride_offset[3] * l;
	    addrs[num++] = addr + tmp[0] + tmp[1] + tmp[2] + tmp[3];
	  }
	}
      }
    }
    break;;
  case 5:
    for(int i=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<array[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int l=0;l<array[3].length;l++){
            tmp[3] = stride_offset[3] * l;
	    for(int m=0;m<array[4].length;m++){
	      tmp[4] = stride_offset[4] * m;
	      addrs[num++] = addr + tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4];
	    }
          }
        }
      }
    }
    break;;
  case 6:
    for(int i=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<array[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int l=0;l<array[3].length;l++){
            tmp[3] = stride_offset[3] * l;
            for(int m=0;m<array[4].length;m++){
              tmp[4] = stride_offset[4] * m;
	      for(int n=0;n<array[5].length;n++){
		tmp[5] = stride_offset[5] * n;
		addrs[num++] = addr + tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5];
	      }
            }
          }
        }
      }
    }
    break;;
  case 7:
    for(int i=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<array[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int l=0;l<array[3].length;l++){
            tmp[3] = stride_offset[3] * l;
            for(int m=0;m<array[4].length;m++){
              tmp[4] = stride_offset[4] * m;
              for(int n=0;n<array[5].length;n++){
                tmp[5] = stride_offset[5] * n;
		for(int p=0;p<array[6].length;p++){
		  tmp[6] = stride_offset[6] * p;
		  addrs[num++] = addr + tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6];
		}
	      }
            }
          }
        }
      }
    }
    break;;
  }
}

