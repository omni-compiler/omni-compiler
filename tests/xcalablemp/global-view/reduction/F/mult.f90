      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
!$xmp nodes p(*)
!$xmp template t(N,N)
!$xmp distribute t(block,*) onto p
      integer a(N), sa, ansa, result
      real*8  b(N), sb, ansb
      real*4  c(N), sc, ansc
!$xmp align a(i) with t(i,*)
!$xmp align b(i) with t(i,*)
!$xmp align c(i) with t(i,*)

      sa = 1
      sb = 1.0
      sc = 1.0
!$xmp loop on t(i,:)
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
      
!$xmp loop on t(i,:)
      do i=1, N
         sa = sa*a(i)
      enddo
!$xmp reduction(*:sa)

!$xmp loop on t(i,:)
      do i=1, N
         sb = sb*b(i)
      enddo
!$xmp reduction(*:sb)

!$xmp loop on t(i,:)
      do i=1, N
         sc = sc*c(i)
      enddo
!$xmp reduction(*:sc)

      ansa = 1024
      ansb = 1024.0
      ansc = 1024.0

      result = 0
      if(  sa .ne. ansa .or. abs(sb-ansb) .gt. 0.00001 .or. abs(sc-ansc) .gt. 0.00001) then
         result = 1  ! ERROR
      endif

!$xmp reduction(+:result)
!$xmp task on p(1)
      if( result .eq. 0 ) then
         write(*,*) "PASS"
      else
         write(*,*) "ERROR"
         call exit(1)
      endif
!$xmp end task
      end
