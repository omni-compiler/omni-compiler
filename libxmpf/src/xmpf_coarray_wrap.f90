function xmpf_coarray_malloc_r4_(datasize) result(ptr)
  integer :: datasize
  real*4 :: data(datasize)
  pointer (ptr, data)

  call xmpf_coarray_malloc_(datasize, 4, ptr)
  return
end function

function xmpf_coarray_malloc_c8_(datasize) result(ptr)
  integer :: datasize
  complex*8 :: data(datasize)
  pointer (ptr, data)

  call xmpf_coarray_malloc_(datasize, 8, ptr)
  return
end function
