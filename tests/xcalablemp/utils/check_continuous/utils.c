#include <stdbool.h>
#define _XMP_N_INT_FALSE false
#define _XMP_N_INT_TRUE  true

typedef struct _XMP_array_section{
  long long start;
  long long length;
  long long stride;
  long long elmts;
  long long distance;
} _XMP_array_section_t;

unsigned int _XMP_get_dim_of_allelmts(const int dims,
                                      const _XMP_array_section_t* array_info)
{
  unsigned int val = dims;

  for(int i=dims-1;i>=0;i--){
    if(array_info[i].start == 0 && array_info[i].length == array_info[i].elmts)
      val--;
    else
      return val;
  }

  return val;
}

int _check_continuous(const _XMP_array_section_t *array_info, const int dims, const int elmts)
{
  // Only 1 elements is transferred.
  // e.g.) a[2]
  if(elmts == 1)
    return _XMP_N_INT_TRUE;

  // Only the last dimension is transferred.
  // e.g.) a[1][2][2:3]
  if(array_info[dims-1].length == elmts && array_info[dims-1].stride == 1)
    return _XMP_N_INT_TRUE;

  // Does non-continuous dimension exist ?
  for(int i=0;i<dims;i++)
    if(array_info[i].stride != 1 && array_info[i].length != 1)
      return _XMP_N_INT_FALSE;

  // (.., d-3, d-2)-th dimension's length is "1" &&
  // d-1-th stride is "1" &&
  // (d, d+1, ..)-th dimensions are ":".
  // e.g.) a[1][3][1:2][:]    // d == 3
  //       a[1][3:2:2][:][:]  // d == 2
  //       a[2][:][:][:]      // d == 1
  //       a[:][:][:][:]      // d == 0

  int d = _XMP_get_dim_of_allelmts(dims, array_info);
  if(d == 0){
    return _XMP_N_INT_TRUE;
  }
  else if(d == 1){
    if(array_info[0].stride == 1)
      return _XMP_N_INT_TRUE;
    else
      return _XMP_N_INT_FALSE;
  }
  else if(d == 2){
    if(array_info[0].length == 1 && array_info[1].stride == 1)
      return _XMP_N_INT_TRUE;
    else
      return _XMP_N_INT_FALSE;
  }
  else if(d == 3){
    if(array_info[0].length == 1 && array_info[1].length == 1 &&
       array_info[2].stride == 1)
      return _XMP_N_INT_TRUE;
    else
      return _XMP_N_INT_FALSE;
  }
  else if(d == 4){
    if(array_info[0].length == 1 && array_info[1].length == 1 &&
       array_info[2].length == 1 && array_info[3].stride == 1)
      return _XMP_N_INT_TRUE;
    else
      return _XMP_N_INT_FALSE;
  }
  else if(d == 5){
    if(array_info[0].length == 1 && array_info[1].length == 1 &&
       array_info[2].length == 1 && array_info[3].length == 1 &&
       array_info[4].stride == 1)
      return _XMP_N_INT_TRUE;
    else
      return _XMP_N_INT_FALSE;
  }
  else if(d == 6){
    if(array_info[0].length == 1 && array_info[1].length == 1 &&
       array_info[2].length == 1 && array_info[3].length == 1 &&
       array_info[4].length == 1 && array_info[5].stride == 1)
      return _XMP_N_INT_TRUE;
    else
      return _XMP_N_INT_FALSE;
  }
  else if(d == 7){
    if(array_info[0].length == 1 && array_info[1].length == 1 &&
       array_info[2].length == 1 && array_info[3].length == 1 &&
       array_info[4].length == 1 && array_info[5].length == 1 &&
       array_info[6].stride == 1)
      return _XMP_N_INT_TRUE;
    else
      return _XMP_N_INT_FALSE;
  }

  return -1; // dummy
}
