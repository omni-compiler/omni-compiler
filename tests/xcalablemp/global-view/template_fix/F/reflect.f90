program tfix

  integer m(4)
  integer, parameter :: N = 120
  integer xmp_node_num
  integer im1

!$xmp nodes p(4)

!$xmp template t(:)
!$xmp distribute t(gblock(*)) onto p

  integer, allocatable ::  a(:)
!$xmp align a(i) with t(i)
!$xmp shadow a(2)

  m = (/ 8, 16, 32, 64 /)

!$xmp template_fix (gblock(m)) t(N)

  allocate (a(N))

!$xmp loop (i) on t(i)
  do i = 1, N
     a(i) = i
  end do

!$xmp reflect (a) width (/periodic/1:1) async (100)
!$xmp wait_async (100)

!$xmp loop (i) on t(i)
  do i = 1, N
     im1 = i - 1
     if (a(i-1) /= mod(im1-1+N, N)+1) then
        write(*,*) "ERROR: Lower in", i, a(i-1), mod(im1-1+N, N)+1
        call exit(1)
     end if
     if (a(i+1) /= mod(im1+1+N, N)+1) then
        write(*,*) "ERROR: Upper in", i, a(i+1), mod(im1+1+N, N)+1
        call exit(1)
   end if
end do

!$xmp task on p(1)
  write(*,*) "PASS"
!$xmp end task

end program tfix
