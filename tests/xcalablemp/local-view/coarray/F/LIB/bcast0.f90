program bcast
!!   include 'xmp_coarray.h'
  integer a(10,20)

  !----------------------- init
  me=this_image()
  a=0
  if (me==1) then
     do i=1,10
        a(i,i)=i
        a(i,i+10)=i
     enddo
  endif
        
  syncall

  !----------------------- exec
  call co_broadcast(a,1)

  !----------------------- check
  nerr=0
  do j=1,20
     do i=1,10
        if (j==i .or. j==i+10) then
           nok=i
        else
           nok=0
        endif
        if (a(i,j)==nok) cycle
        nerr=nerr+1
        print '("[",i0,"] a(",i0,",",i0,") should be ",i0," but ",i0)', &
             me, i, j, nok, a(i,j)
     enddo
  enddo

  !----------------------- output
  if (nerr==0) then 
     print '("[",i0,"] OK")', me
  else
     print '("[",i0,"] number of NGs: ",i0)', me, nerr
  end if

  end
