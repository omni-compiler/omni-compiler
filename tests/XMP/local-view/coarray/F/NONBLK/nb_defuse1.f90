! This program requires blocking communication to execute correctly.

  program nb_defuse1

    complex*8 a(120,120), b(120,120)[*], c(120,120)
    integer nerr

    me = this_image()

    do j=1,120
       do i=1,120
          a(i,j) = cmplx(i,j)
       end do
    enddo
    b=(4.0,4.0)
    c=(1.0,1.0)

    sync all

    !---------------------------- execution
    if (me==3) then
       b(41:80,:)[1] = a(41:80,:)*2       !(a)
       c(:,41:80) = b(:,41:80)[1];        !(b)
    endif

    sync all

    !---------------------------- check and output start
    nerr = 0
    do j=1,120
       do i=1,120
          !!----- effect of (b)
          if (me==3.and.j>=41.and.j<=80) then
             if (i>=41.and.i<=80) then
                if ((c(i,j).ne.a(i,j)*2)) then
                   write(*,102) i,j,me,c(i,j),a(i,j)*2
                   nerr=nerr+1
                endif
             else
                if (c(i,j).ne.(4.0,4.0)) then
                   write(*,102) i,j,me,c(i,j),(4.0,4.0)
                   nerr=nerr+1
                endif
             endif
          else
             if (c(i,j).ne.(1.0,1.0)) then
                write(*,102) i,j,me,c(i,j),(1.0,1.0)
                nerr=nerr+1
             endif
          endif

       enddo
    end do

    if (nerr==0) then 
       print '("result[",i0,"] OK")', me
    else
       print '("result[",i0,"] number of NGs: ",i0)', me, nerr
       call exit(1)
    end if
    !---------------------------- check and output end

101 format ("b(",i0,",",i0,")[",i0,"]=(",f8.1,",",f8.1,") should be (",f8.1,",",f8.1,")")
102 format ("c(",i0,",",i0,")[",i0,"]=(",f8.1,",",f8.1,") should be (",f8.1,",",f8.1,")")

  end program
