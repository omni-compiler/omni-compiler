program reduct
!!   include 'xmp_coarray.h'
  integer,parameter:: N=3
  integer a(N,N)[*]
  integer*8 q(N),q0(N),qsum(N),qmaxmin(0:1,N)
  integer correct

  !----------------------- init
  me=this_image()
  qsum=0
  qmaxmin=0

  if (me==3) then
     !!     me   1  2  3
     a(1,:) = (/ 4, 9, 2/)
     a(2,:) = (/ 3, 5, 7/)  
     a(3,:) = (/ 8, 1, 6/)
  end if
  call co_broadcast(a,3)
  q0(:) = a(:,me)

  !----------------------- exec
  q = q0
  call co_sum(q, result_image=1_4)
  qsum = q
  q = q0
  call co_max(result_image=2, source = q)
  qmaxmin(0,:) = q
  q = q0
  call co_min(source=q, result_image=3)
  qmaxmin(1,:) = q

  !----------------------- check
  nerr=0

  if (me==1) then
     do i=1,N
        correct = sum(a(i,:))
        if (qsum(i).ne.correct) then
           nerr=nerr+1
           print 111, me, "qsum(", i, correct, qsum(i)
        endif
     enddo
  endif

  if (me==2) then
     do i=1,N
        correct = maxval(a(i,:))
        if (qmaxmin(0,i).ne.correct) then
           nerr=nerr+1
           print 111, me, "qmaxmin(0,", i, correct, int(qmaxmin(0,i))
        endif
     enddo
  endif

  if (me==3) then
     do i=1,N
        correct = minval(a(i,:))
        if (qmaxmin(1,i).ne.correct) then
           nerr=nerr+1
           print 111, me, "qmaxmin(1,", i, correct, int(qmaxmin(1,i))
        endif
     enddo
  endif

111 format ("[",i0,"] ",a,i0,") must be ",i0," but ",i0,".")

  !----------------------- output
  if (nerr==0) then 
     print '("[",i0,"] OK")', me
  else
     print '("[",i0,"] number of NGs: ",i0)', me, nerr
  end if

  end
