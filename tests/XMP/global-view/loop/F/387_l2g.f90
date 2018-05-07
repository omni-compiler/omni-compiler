  integer,parameter :: n1=10, n2=20
!$xmp nodes p(1)
!$xmp template t1(n1,n1)
!$xmp template t2(n2,n2)
!$xmp distribute t1(*,block) onto p
!$xmp distribute t2(*,block) onto p
  integer :: a(n2,n2)
!$xmp align a(i,j) with t2(i,j)

!$xmp loop (i,j) on t2(i,j)
  do j=1,n2
    do i=1,n2
      a(i,j)=0
    end do
  end do

!$xmp loop (i,j) on t1(i,j)
  do j=1,n1
    jj=2*j
    do i=1,n1
      ii=2*i
      a(ii,jj)=ii+jj
    end do
  end do

ierr=0
!$xmp loop (i,j) on t2(i,j)
  do j=2,n2,2
    do i=2,n2,2
     ierr=ierr+abs(a(i,j)-i-j)
    end do
  end do

  if (ierr .eq. 0) then
    print *, 'PASS'
  else
    print *, 'ERROR'
    call abort
  end if

  end
