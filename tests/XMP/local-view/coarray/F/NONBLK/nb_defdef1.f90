! This program requires blocking communication to execute correctly.

  program nb_defdef1

    complex*8 a(120,120), b(120,120)[*]
    integer nerr

    me = this_image()

    do j=1,120
       do i=1,120
          a(i,j) = cmplx(i,j)
       end do
    enddo
    b=(0.0,0.0)

    sync all

    !---------------------------- execution
    if (me==3) then
       b(41:80,:)[1] = a(41:80,:)*2
       b(:,41:80)[1] = a(:,41:80)*4
    endif

    sync all

    !---------------------------- check and output start
    nerr = 0
    do j=1,120
       do i=1,120
          if (me==1.and.j>=41.and.j<=80) then
             if (b(i,j).ne.a(i,j)*4) then
                write(*,101) i,j,me,b(i,j),a(i,j)*4
                nerr=nerr+1
             endif
          else if (me==1.and.i>=41.and.i<=80) then
             if (b(i,j).ne.a(i,j)*2) then
                write(*,101) i,j,me,b(i,j),a(i,j)*2
                nerr=nerr+1
             endif
          else
             if (b(i,j).ne.(0.0,0.0)) then
                write(*,101) i,j,me,b(i,j),(0.0,0.0)
                nerr=nerr+1
             endif
          endif
       end do
    end do

    if (nerr==0) then 
       print '("result[",i0,"] OK")', me
    else
       print '("result[",i0,"] number of NGs: ",i0)', me, nerr
    end if
    !---------------------------- check and output end

101 format ("b(",i0,",",i0,")[",i0,"]=(",f8.1,",",f8.1,") should be (",f8.1,",",f8.1,")")

  end program
