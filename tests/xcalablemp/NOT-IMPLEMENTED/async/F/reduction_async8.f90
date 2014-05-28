! testp041.f
! reduction指示文(min)のテスト

      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
      integer random_array(N), ans_val, val
!$xmp nodes p(*)
!$xmp template t(N)
!$xmp distribute t(block) onto p
      character(len=2) result

      result = 'OK'
      do k=114, 10000, 17
         random_array(1) = k
         do i=2, N
            random_array(i) =
     $           mod(random_array(i-1)**2, 100000000)
            random_array(i) =
     $           mod((random_array(i)-mod(random_array(i),100))/100,
     $           10000)
         enddo

         val = 2147483647
!$xmp loop on t(i)
         do i=1, N
            val = min(val, random_array(i))
         enddo
!$xmp reduction(min: val) async(1)
         ans_val = 2147483647
         do i=1, N
            ans_val = min(ans_val, random_array(i))
         enddo
!$xmp wait_async(1)

         if(val .ne. ans_val) then
            print *, val, '!=', ans_val
            result = 'NG'
         endif
      enddo

      print *, xmp_node_num(), 'testp041.f ', result

      end
