program reduct
!!   include 'xmp_coarray.h'
  integer,parameter:: N=3
  integer a(N,N)[*]
  integer*8 q(N),qsum(N),qmaxmin(0:1,N)
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
  q(:) = a(:,me)

  !----------------------- exec
  !! restriction of FrontEnd: It can recognize 1_4 as int*4 but cannot 2.
  call co_sum(q, qsum, 1_4)
  call co_max(q, result_image=2, result=qmaxmin(0,:))
  call co_min(result=qmaxmin(1,:), source=q, result_image=3)

  !----------------------- check
  nerr=0

  do i=1,N
     if (me==1) then
        correct = sum(a(i,:))
     else
        correct = 0
     endif
     if (qsum(i).ne.correct) then
        nerr=nerr+1
        print 111, me, "qsum(", i, correct, qsum(i)
     endif
  enddo

  do i=1,N
     if (me==2) then
        correct = maxval(a(i,:))
     else
        correct = 0
     endif
     if (qmaxmin(0,i).ne.correct) then
        nerr=nerr+1
        print 111, me, "qmaxmin(0,", i, correct, int(qmaxmin(0,i))
     endif
  enddo

  do i=1,N
     if (me==3) then
        correct = minval(a(i,:))
     else
        correct = 0
     endif
     if (qmaxmin(1,i).ne.correct) then
        nerr=nerr+1
        print 111, me, "qmaxmin(1,", i, correct, int(qmaxmin(1,i))
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
