  program main
    !$xmp nodes p(8)
    !$xmp nodes q1(3)=p(1:3)
    !$xmp nodes q2(4)=p(4:7)

    real as(10,10)[*]

    integer,allocatable:: ad(:,:)[:]

    allocate (ad(10,10)[*])

    nerr=0
    !$xmp tasks
      !$xmp task on q1
        call sub1(nerr)
      !$xmp end task
      !$xmp task on q2
        call sub2(nerr,ad)
      !$xmp end task
    !$xmp end tasks

    if (nerr==0) then 
       print '("[",i0,"] OK")', this_image()
    else
       print '("[",i0,"] number of NGs: ",i0)', this_image(), nerr
    end if

  contains

  subroutine sub1(nerr)

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
    call co_sum(ad1, ad1sum)

    if (ad1sum(3,4).ne.3*4*(1+2+3)) then
       nerr=nerr+1
    endif

  end subroutine sub1

  subroutine sub2(nerr,ad2)

    integer,allocatable:: ad2(:,:)[:]
    !$xmp coarray on p :: ad2
                               !! whole nodes

    if (me==1) sync images(2)
    if (me==2) sync images(1)

    if (this_image()==1) then
       do k=1,8
          do j=1,10
             do i=1,10
                ad2(i,j)[k]=i*j*k
             enddo
          enddo
       enddo
    end if

    if (ad2(1,1).ne.1*1*(this_image()+3)) then
       write(*,200) this_image(), "ad2(1,1)", 1*1*(this_image()+3), ad2(1,1)
       nerr=nerr+1
    endif

200 format("q2(",i0,"): ",a," should be ",i0," but ",i0,".")

  end subroutine sub2

  end 
