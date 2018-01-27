  integer, parameter :: mimax=129
  integer, parameter :: mjmax=65
  integer, parameter :: mkmax=65

  !$xmp nodes nd(2,2,2)
  !$xmp template t(mimax,mjmax,mkmax)
  !$xmp distribute t(block,block,block) onto nd

  real(4) :: p(mimax,mjmax,mkmax)
  !$xmp align (i,j,k) with t(i,j,k) :: p

  !$xmp array on t(:,:,:)
  p= 0.0

  !$xmp array on t(2:mimax-1,2:mjmax-1,2:mkmax-1)
  p(2:mimax-1,2:mjmax-1,2:mkmax-1)= 1.0

  nerr=0
  !$xmp loop (i,j,k) on t(i,j,k)
  do k=1,3
     do j=1,3
        do i=1,3
           if (i==1.or.j==1.or.k==1) then
              value=0.0
           else 
              value=1.0
           endif
           if (p(i,j,k).ne.value) then
              nerr=nerr+1
           endif
        enddo
     enddo
  enddo

  !$xmp task on nd(1,1,1)
  if (nerr==0) then
     write(*,*) "PASS"
  else
     write(*,*) "NG nerr=",nerr
  endif
  !$xmp end task

end
