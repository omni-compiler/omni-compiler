! testp095.f
! loop指示文とreduction指示文のテスト (1次元では意味なし？)

      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
!$xmp nodes p(4)
!$xmp template t(N,N,N)
!$xmp distribute t(*,*,gblock((/333,555,111,1/))) onto p
      integer a1(N), a2(N), aa
!$xmp align a1(i) with t(*,*,i)
!$xmp align a2(i) with t(*,*,i)
      character(len=2) result

!$xmp loop on t(:,:,i)
      do i=1, N
         a1(i) = 0
         a2(i) = i
      enddo
      
      result = 'OK'
!$xmp loop on t(:,:,i)
      do i=1, N
         aa = a2(i)
!$xmp reduction(ior: aa) async(1)
         if(a1(i) .ne. 0) result = 'NG'
!$xmp wait_async(1)
         aa = aa-1
         a1(i) = aa
      enddo

!$xmp loop on t(:,:,i)
      do i=1, N
         if(a1(i) .ne. i-1) result = 'NG'
      enddo

      print *, xmp_node_num(), 'testp095.f ', result

      end
      
