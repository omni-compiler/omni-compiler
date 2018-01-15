! testp029.f
! reduction指示文(*)のテスト

      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
!$xmp nodes p(*)
!$xmp template t(N,N,N)
!$xmp distribute t(*,*,block) onto p
      integer a(N), sa, ansa
      real*8  b(N), sb, ansb
      real*4  c(N), sc, ansc
      character(len=2) result
!$xmp align a(i) with t(*,*,i)
!$xmp align b(i) with t(*,*,i)
!$xmp align c(i) with t(*,*,i)

      sa = 1
      sb = 1.0
      sc = 1.0
!$xmp loop on t(:,:,i)
      do i=1, N
         if(mod(i,100) .eq. 0) then
            a(i) = 2
            b(i) = 2.0
            c(i) = 2.0
         else
            a(i) = 1
            b(i) = 1.0
            c(i) = 1.0
         endif
      enddo
      
!$xmp loop on t(:,:,i)
      do i=1, N
         sa = sa*a(i)
      enddo

!$xmp reduction(*:sa) async(1)

!$xmp loop on t(:,:,i)
      do i=1, N
         sb = sb*b(i)
      enddo
!$xmp wait_async(1)
      sa = sa*3

!$xmp reduction(*:sb) async(1)

!$xmp wait_async(1)
      sb = sb*1.5

!$xmp loop on t(:,:,i)
      do i=1, N
         sc = sc*c(i)
      enddo

!$xmp reduction(*:sc) async(1)

!$xmp wait_async(1)
      sc = sc*0.5

      ansa = 1024*3
      ansb = 1024.0*1.5
      ansc = 1024.0*0.5

      result = 'OK'
      if(  sa .ne. ansa .or.
     $     abs(sb-ansb) .gt. 0.00001 .or.
     $     abs(sc-ansc) .gt. 0.00001) then
         result = 'NG'
      endif

      print *, xmp_node_num(), 'testp029.f ', result

      end

      
