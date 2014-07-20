! testp089.f
! loop指示文とarray指示文のテスト

      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
!$xmp nodes p(*)
!$xmp template t1(N,N,N)
!$xmp template t2(N,N,N)
!$xmp template t3(N,N,N)
!$xmp distribute t1(*,*,gblock((/325,435,111,129/))) onto p
!$xmp distribute t2(*,gblock((/325,435,111,129/)),*) onto p
!$xmp distribute t3(gblock((/325,435,111,129/)),*,*) onto p
      integer a(N)
      real*8  b(N)
      real*4  c(N)
!$xmp distribute a(i) with t1(*,*,i)
!$xmp distribute b(i) with t2(*,i,*)
!$xmp distribute c(i) with t3(i,*,*)
      character(len=2) result

      result = 'OK'
!$xmp loop on t1(:,:,i)
      do i=1, N
!$xmp array on t1(:,:,i)
         a(i) = i
      enddo
!$xmp loop on t2(:,i,:)
      do i=1, N
!$xmp array on t2(:,i,:)
         b(i) = dble(i)
      enddo
!$xmp loop on t3(i,:,:)
      do i=1, N
!$xmp array on t3(i,:,:)
         c(i) = real(i)
      enddo

!$xmp loop on t1(:,:,i)
      do i=1, N
         if(a(i) .ne. i) result = 'NG'
      enddo
!$xmp loop on t2(:,i,:)
      do i=1, N
         if(b(i) .ne. dble(i)) result = 'NG'
      enddo
!$xmp loop on t3(i,:,:)
      do i=1, N
         if(c(i) .ne. real(i)) result = 'NG'
      enddo

      print *, xmp_node_num(), 'testp089.f ', result

      end
      
