program bcast_arg2i8
  integer a(10),b(10),c(10)
  integer, parameter :: n=2
!!  integer(kind=n**3) nn
  integer(kind=n) nn
!!  integer(kind=8) nn

  !----------------------- init
  me=this_image()
  if (me==2) then
     do i=1,10
        a(i)=i
        b(i)=i
        c(i)=i
     enddo
  else
     do i=1,10
        a(i)=0
        b(i)=0
        c(i)=0
     enddo
  endif
        
  syncall

  !----------------------- exec
  call co_broadcast(a,int(2,8))
  call co_broadcast(b,source_image=1+1_8)
  nn = 2_2
  call co_broadcast(source_image=nn,source=c)

  !----------------------- check
  nerr=0
  do i=1,10
     nok=i
     if (a(i) /= nok) then
        nerr=nerr+1
        print '("[",i0,"] a(",i0,") should be ",i0," but ",i0)', &
             me, i, nok, a(i)
     end if
     if (b(i) /= nok) then
        nerr=nerr+1
        print '("[",i0,"] b(",i0,") should be ",i0," but ",i0)', &
             me, i, nok, b(i)
     end if
     if (c(i) /= nok) then
        nerr=nerr+1
        print '("[",i0,"] c(",i0,") should be ",i0," but ",i0)', &
             me, i, nok, c(i)
     end if
  enddo

  !----------------------- output
  if (nerr==0) then 
     print '("[",i0,"] OK")', me
  else
     print '("[",i0,"] number of NGs: ",i0)', me, nerr
  end if

  end
