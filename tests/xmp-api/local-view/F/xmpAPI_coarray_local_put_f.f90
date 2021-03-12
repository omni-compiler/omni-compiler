! coarray - local array 
program coarray_local_put_test
  use xmp_api
  integer, parameter :: SIZE=10, DIMS=1
  integer(4), pointer :: a ( : ) => null ( )
  integer(4), allocatable :: b(:)

  integer i,j
  integer(8) :: a_desc, b_local_desc
  integer(8), dimension(DIMS) :: a_lb,a_ub,b_lb, b_ub
  integer(4) :: img_dims(1)
  integer(8) :: a_sec, b_local_sec
  integer(4) status
  integer(4) :: my_image

  call xmp_api_init

  allocate(b(SIZE))

  a_lb(1) = 1
  a_ub(1) = SIZE

  ! print *,"Setting array a"
  call xmp_new_coarray(a_desc, 4, DIMS, a_lb, a_ub, 1, img_dims)
  call xmp_coarray_bind(a_desc,a)
  call xmp_new_array_section(a_sec,1)

  b_lb(1) = 1
  b_ub(1) = SIZE

  ! print *,"Setting array local b"
  call xmp_new_local_array(b_local_desc,4,DIMS,b_lb,b_ub,loc(b))
  call xmp_new_array_section(b_local_sec,1)

  do i=1,SIZE
     a(i) = 0
  enddo
  do i=1,SIZE
     b(i) = i
  enddo

  my_image = xmp_this_image()

  call xmp_sync_all(status)

  if(my_image == 1) then
    call xmp_array_section_set_triplet(a_sec, &
         1,int(1,kind=8),int(8,kind=8),1,status)
    call xmp_array_section_set_triplet(b_local_sec, &
         1,int(1,kind=8),int(8,kind=8),1,status)

    img_dims(1) = 2
    ! local array b of image 1 -> coarray a of image 2
    ! print *,"xmp_coarray_put_local"
    call xmp_coarray_put_local(img_dims,a_desc,a_sec, &
      b_local_desc,b_local_sec,status) ! local put
  endif

  call xmp_sync_all(status)

  if(my_image == 2) then
    if((a(3) == 3).and.(a(5) == 5).and.(a(7) == 7)) then
      print *," PASS "
   else
      print *," ERROR "
    endif
  endif

  call xmp_free_array_section(a_sec)
  call xmp_free_array_section(b_local_sec)

  call xmp_coarray_deallocate(a_desc, status)
  call xmp_free_local_array(b_local_desc)

  call xmp_api_finalize
end program coarray_local_put_test
