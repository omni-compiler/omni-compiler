! testp066.f
! task指示文とarray指示文の組み合わせ

      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
!$xmp nodes p(4)
!$xmp template t(N,N,N)
!$xmp distribute t(*,*,gblock((/200,100,400,300/))) onto p
      integer a(N), sa, ansa
      real*8  b(N), sb, ansb
      character(len=2) result
!$xmp align a(i) with t(*,*,i)
!$xmp align b(i) with t(*,*,i)
!$xmp shadow a(1:0)
!$xmp shadow b(0:1)

      sa = 0
      sb = 0.0

!$xmp loop on t(:,:,i)
      do i=1, N
         a(i) = 1
         b(i) = 1.0
      enddo

!$xmp task on p(1)
!$xmp array on t(:,:,:)
      a(1:200) = 2
!$xmp array on t(:,:,:)
      b(1:200) = 1.5
!$xmp end task      
      
!$xmp loop on t(:,:,i)
      do i=1, N
         sa = sa+a(i)
         sb = sb+b(i)
      enddo

!$xmp reduction(+:sa)
!$xmp reduction(+:sb)

      ansa = N+200
      ansb = dble(N)+0.5*200

      result = 'OK'
      if(sa .ne. ansa) then
         result = 'NG'
      endif
      if(abs(sb-ansb) .gt. 0.00000000001) then
         result = 'NG'
      endif

      print *, xmp_node_num(), 'testp066.f ', result

      end
