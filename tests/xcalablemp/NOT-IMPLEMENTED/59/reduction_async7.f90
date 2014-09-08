! testp039.f
! reduction指示文(max)のテスト

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

         val = 0
!$xmp loop on t(i)
         do i=1, N
            val = max(val, random_array(i))
         enddo
!$xmp reduction(max: val) async(1)
         ans_val = 0
         do i=1, N
            ans_val = max(ans_val, random_array(i))
         enddo
!$xmp wait_async(1)

         if(val .ne. ans_val) then
            print *, val, '!=', ans_val
            result = 'NG'
         endif
      enddo

      print *, xmp_node_num(), 'testp039.f ', result

      end
