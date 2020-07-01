  module declmod
    !$xmp nodes p(8)
    !$xmp nodes q1(3)=p(1:3)
    !$xmp nodes q2(4)=p(4:7)
  end module declmod

  subroutine sub1(nerr)
    use declmod
    integer,allocatable:: ad1(:,:)[:]
    !$xmp coarray on q1 :: ad1
                               !! no meaning: q1 matches the context
    integer:: ad1sum(10,10)

    allocate (ad1(10,10)[*])

    do j=1,10
       do i=1,10
          ad1(i,j)=i*j*this_image()
       enddo
    enddo

    !$xmp image(p)
    syncall

    call co_sum(ad1, ad1sum)

    if (ad1sum(3,4).ne.3*4*(1+2+3)) then
       nerr=nerr+1
    endif

  end subroutine sub1

  subroutine sub2(nerr)
    use declmod
    integer,allocatable:: ad2(:,:)[:]
    !$xmp coarray on q2 :: ad2
                               !! no meaning: q2 matches the context
    integer:: ad2sum(10,10)

    if (me==1) sync images(2)
    if (me==2) sync images(1)

    allocate (ad2(10,10)[*])

    do j=1,10
       do i=1,10
          ad2(i,j)=i*j*this_image()
       enddo
    enddo

    !$xmp image(p)
    syncall

    call co_sum(ad2, ad2sum)

    if (ad2sum(3,4).ne.3*4*(1+2+3+4)) then
       nerr=nerr+1
    endif

  end subroutine sub2



  program main
    use declmod
    real as(10,10)[*]

    nerr=0
    !$xmp tasks
      !$xmp task on q1
        call sub1(nerr)
      !$xmp end task
      !$xmp task on q2
        call sub2(nerr)
      !$xmp end task
      !$xmp task on p(8)
        !$xmp image (p)
        syncall
      !$xmp end task
    !$xmp end tasks

  if (nerr==0) then 
     print '("[",i0,"] OK")', this_image()
  else
     print '("[",i0,"] number of NGs: ",i0)', this_image(), nerr
     call exit(1)
  end if

  end 


