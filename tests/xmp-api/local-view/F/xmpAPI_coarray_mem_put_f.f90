program coarray_coarray_mem_put_f
  use xmp_api
  integer, parameter :: SIZE=10, DIMS=1
  integer(8) :: s_desc
  character(len=SIZE), pointer :: s => null ()
  character(len=SIZE)::  s1, s2
  integer(8) :: v_desc
  integer(4), pointer :: v => null()
  integer(4) :: v1, v2
  integer(4) :: img_dims(1)
  integer(4) :: my_image, status

  call xmp_api_init

  v1 = 100
  v2 = 200
  s1 = "123456789"
  s2 = "abcdefghi"

  call xmp_new_coarray_mem(v_desc, 4, 1, img_dims)
  call xmp_coarray_bind(v_desc, v)
  v = 200

  call xmp_new_coarray_mem(s_desc, SIZE, 1, img_dims)
  call xmp_coarray_bind(s_desc, s)
  s = "abcdefghi"

  call xmp_sync_all(status)
  
  my_image = xmp_this_image()
  
  if(my_image == 1) then
     img_dims(1) = 2
     call xmp_coarray_mem_put(img_dims,v_desc,v1, status)
     call xmp_coarray_mem_put(img_dims,s_desc,s1,5,status)
  end if

  call xmp_sync_all(status)

  if(my_image == 2) then
!     print *,"v=", v
!     print *,"s=", s
     if(v == v1 .and. s(1:5) == s1(1:5)) then
        print *,'PASS'
     else
        print *,'ERROR'
     end if
  end if

  call xmp_coarray_deallocate(v_desc, status)
  call xmp_coarray_deallocate(s_desc, status)

  call xmp_api_finalize
end program coarray_coarray_mem_put_f
