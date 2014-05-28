! testp037.f
! reduction指示文(ieor)のテスト

      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
      integer random_array(N), ans_val
      integer a(N), sa
!$xmp nodes p(*)
!$xmp template t(N)
!$xmp distribute t(block) onto p
!$xmp align a(i) with t(i)
      character(len=2) result

      result = 'OK'
      do k=114, 10000, 113
         random_array(1) = k
         do i=2, N
            random_array(i) =
     $           mod(random_array(i-1)**2, 100000000)
            random_array(i) =
     $           mod(random_array(i)/100, 10000)
         enddo

!$xmp loop (i) on t(i)
         do i=1, N
            a(i) = random_array(i)
         enddo
         
         sa = 0
!$xmp loop (i) on t(i)
         do i=1, N
            sa = ieor(sa, a(i))
         enddo
!$xmp reduction(ieor: sa) async(1)

         ans_val = 0
         do i=1, N
            ans_val = ieor(ans_val, random_array(i))
         enddo

!$xmp wait_async(1)

         if(  sa .ne. ans_val) then
            result = 'NG'
         endif
      enddo

      print *, xmp_node_num(), 'testp037.f ', result

      end
