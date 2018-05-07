program reduct
!!   include 'xmp_coarray.h'
  integer,parameter:: N=3
  integer a(N,N)[*]
  integer*8 q(N),qsum(N),qmaxmin(0:1,N)

  !----------------------- init
  me=this_image()
  if (me==3) then
     !!    me = 1 2 3
     a(1,:) = (/4,9,2/)
     a(2,:) = (/3,5,7/)
     a(3,:) = (/8,1,6/)
  end if
  call co_broadcast(a,3)

  q(:) = a(:,me)

  !----------------------- exec
  call co_sum(q, qsum)
  call co_max(q, qmaxmin(0,:))
  call co_min(q, qmaxmin(1,:))

  !----------------------- check
  nerr=0

  do i=1,N
!!     print *,"i,qsum(i)=",i,qsum(i)
     if (qsum(i).ne.sum(a(i,:))) then
        nerr=nerr+1
        print 111, me, "qsum(", i, a(i,me), qsum(i)
     endif
  enddo

  do i=1,N
!!     print *,"i,qmaxmin(0,i)=",i,qmaxmin(0,i)
     if (qmaxmin(0,i).ne.maxval(a(i,:))) then
        nerr=nerr+1
        print 111, me, "qmaxmin(0,", i, int(a(i,me)), int(qmaxmin(0,i))
     endif
  enddo

  do i=1,N
!!     print *,"i,qmaxmin(1,i)=",i,qmaxmin(1,i)
     if (qmaxmin(1,i).ne.minval(a(i,:))) then
        nerr=nerr+1
        write(*,111) me, "qmaxmin(1,", i, int(a(i,me)), int(qmaxmin(1,i))
     endif
  enddo

111 format ("[",i0,"] ",a,i0,") must be ",i0," but ",i0,".")

  !----------------------- output
  if (nerr==0) then 
     print '("[",i0,"] OK")', me
  else
     print '("[",i0,"] number of NGs: ",i0)', me, nerr
  end if

  end
