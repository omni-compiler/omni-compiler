program coarray_get_2dims_to_1dim
  use xmp_api
  integer, parameter :: SIZE=10, DIMS=2
!  integer :: a(SIZE,SIZE)[*], b(SIZE,SIZE)[*]
  integer , POINTER :: a ( : , : ) => null ( )
  integer , POINTER :: b ( : , : ) => null ( )

  integer(8) :: start1, start2, end1, end2
  integer(4) ::  length1, length2, stride1, stride2
  integer i,j, pos1, pos2, ret
  integer(8) :: a_desc, b_desc
  integer(8), dimension(DIMS) :: a_lb,a_ub,b_lb, b_ub
  integer(4) :: img_dims(1)
  integer(8) :: a_sec, b_sec
  integer(4) status

  call xmp_api_init
!  print *,'xmp_api_init done ...'

  a_lb(1) = 1
  a_lb(2) = 1
  a_ub(1) = SIZE
  a_ub(2) = SIZE

  b_lb(1) = 1
  b_lb(2) = 1
  b_ub(1) = SIZE
  b_ub(2) = SIZE
  call xmp_new_coarray(a_desc, 4, DIMS, a_lb, a_ub, 1, img_dims)
  call xmp_new_coarray(b_desc, 4, DIMS, b_lb, b_ub, 1, img_dims)
!  print *,'xmp_new_coarray done ...'

  call xmp_coarray_bind(a_desc,a)
  call xmp_coarray_bind(b_desc,b)
!  print *,'xmp_coarray_bind done ...'
  
  ret = 0
  DO I = 1, SIZE
     DO J = 1, SIZE
        b(I,J) = xmp_this_image()
     end DO
  end DO

  DO I = 1, SIZE
    DO J = 1, SIZE
      a(I,J) = -1
    end DO
  end DO
!  sync all
  call xmp_sync_all(status)

! coarray_get subroutine
  call get1(b_desc, a_desc, SIZE*SIZE)

  call xmp_coarray_deallocate(a_desc, status)
  call xmp_coarray_deallocate(b_desc, status)

  call xmp_api_finalize

end program coarray_get_2dims_to_1dim

subroutine get1(src_desc, dst_desc, bufsize)
  use xmp_api
  integer(8) :: src_desc, dst_desc
  integer(4) bufsize

  integer , POINTER :: bufsnd ( : ) => null ( )
  integer , POINTER :: bufrcv ( : ) => null ( )

  integer i,ret
  integer(8) :: snd_desc, rcv_desc
  integer(8) :: snd_sec, rcv_sec
  integer(8) :: start1, end1
  integer(4) ::  length1, stride1
  integer(8), dimension(1) :: snd_lb,snd_ub,rcv_lb, rcv_ub
  integer(4) :: img_dims(1)
  integer(4) status

  ret = 0

  snd_lb(1) = 1
  snd_ub(1) = bufsize
  rcv_lb(1) = 1
  rcv_ub(1) = bufsize

  call xmp_reshape_coarray(snd_desc, src_desc, 4, 1, snd_lb, snd_ub, 1, img_dims)
  call xmp_reshape_coarray(rcv_desc, dst_desc, 4, 1, rcv_lb, rcv_ub, 1, img_dims)
!  snd_desc = xmp_reshape_coarray(src_desc, sizeof(int), 1,dims,1,img_dims,(void **)&snd_p);
!  rcv_desc = xmp_reshape_coarray(dst_desc, sizeof(int), 1,dims,1,img_dims,(void **)&rcv_p);

  call xmp_coarray_bind(snd_desc,bufsnd)
  call xmp_coarray_bind(rcv_desc,bufrcv)

  if(xmp_this_image() == 1) then
     call xmp_new_array_section(snd_sec,1)
     call xmp_new_array_section(rcv_sec,1)

     !  bufrcv(start1:end1:stride1) = bufsnd(start1:end1:stride1)[2]
     start1 = 1
     end1 = bufsize
     stride1 = 1
     call xmp_array_section_set_triplet(snd_sec,1,start1,end1,stride1,status)
     call xmp_array_section_set_triplet(rcv_sec,1,start1,end1,stride1,status)

     img_dims(1) = 2
     call xmp_coarray_get(img_dims,snd_desc,snd_sec,rcv_desc,rcv_sec,status)

     call xmp_free_array_section(snd_sec)
     call xmp_free_array_section(rcv_sec)
  end if

  DO I = 1, bufsize
    if(xmp_this_image() == 1) then
      if (bufrcv(I) /= 2) then
        print *, I, bufrcv(I)
        ret = -1
        goto 10
      end if
    endif
  end DO
10 call xmp_sync_all(status)


  if(xmp_this_image() == 1) then
     if(ret == 0) then
        print *,'PASS'
     else
        print *,'ERROR'
     end if
  end if
 
end

