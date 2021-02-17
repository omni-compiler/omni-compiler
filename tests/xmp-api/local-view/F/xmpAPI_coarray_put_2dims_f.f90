program coarray_put_2dims_f
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

!  CALL xmpf_init_all_ ( )
  call xmp_init_all
!  print *,'xmp_init_all done ...'

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

!  print *,'xmp_corray_bind done ...'
  
  ret = 0
  DO I = 1, SIZE
     DO J = 1, SIZE
        b(I,J) = xmp_this_image()
     end DO
  end DO

  call xmp_new_array_section(a_sec,2)
  call xmp_new_array_section(b_sec,2)
  
  DO start1 = 1, 2
     DO length1 = 1, SIZE
        DO stride1 = 1, 3
           DO start2 = 1,2
              DO length2 = 1, SIZE
                 DO stride2 = 1, 3
                    end1 = start1+length1*stride1-1
                    end2 = start2+length2*stride2-1
                    if(end1 <= SIZE .and. end2 <= SIZE) then
                       DO I = 1, SIZE
                          DO J = 1, SIZE
                             a(I,J) = -1
                          end DO
                       end DO

!                       sync all
                       call xmp_sync_all(status)

                       if(xmp_this_image() == 1) then
!                          a(start1:end1:stride1,start2:end2:stride2) = &
!                          b(start1:end1:stride1,start2:end2:stride2)[2]
                          call xmp_array_section_set_triplet(a_sec,1,start1,end1,stride1,status)
                          call xmp_array_section_set_triplet(a_sec,2,start2,end2,stride2,status)
                          call xmp_array_section_set_triplet(b_sec,1,start1,end1,stride1,status)
                          call xmp_array_section_set_triplet(b_sec,2,start2,end2,stride2,status)
                          img_dims(1) = 2;
                          call xmp_coarray_put(img_dims,a_desc,a_sec,b_desc,b_sec,status);
                       end if
                       
                       call xmp_sync_all(status)
                       
                       if(xmp_this_image() == 2) then
                          do i= 0, length1-1
                             do j= 0, length2-1
                                pos1 = start1 + i*stride1
                                pos2 = start2 + j*stride2
                                if (a(pos1,pos2) /= 1) then
!                                   print '"a(",i0,":",i0,":",i0,",",i0,":",i0,":",i0,")"', &
                                   print *, a(pos1,pos2), start1,length1,stride1,start2,length2,stride2
                                   ret = -1
                                   goto 10
                                end if
                             end do
                          end do
                       end if
                    end if
                 end DO
              end DO
           end DO
        end DO
     end DO
  end DO
!10  sync all
10 call xmp_sync_all(status)

  call xmp_free_array_section(a_sec)
  call xmp_free_array_section(b_sec)

  if(xmp_this_image() == 2) then
     if(ret == 0) then
        print *,'PASS'
     else
        print *,'ERROR'
     end if
  end if

  call xmp_coarray_deallocate(a_desc, status)
  call xmp_coarray_deallocate(b_desc, status)

! CALL xmpf_finalize_all_ ( )
  call xmp_finalize_all

end program coarray_put_2dims_f

