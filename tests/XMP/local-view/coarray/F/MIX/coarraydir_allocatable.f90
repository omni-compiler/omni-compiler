  !$xmp nodes p(8)
  !$xmp nodes q1(3)=p(1:3)
  !$xmp nodes q2(2,2)=p(4:7)

  real as(10,10)[*]
  real,allocatable:: ad2(:,:)[:],ad1(:,:)[:]
  !$xmp coarray on q1 :: ad1
!!  !$xmp coarray on q2 :: ad2


      !$xmp tasks
      !$xmp task on q1
!!  allocate(ad1(1:10,1:10)[*])
!!  ad1=this_image()
!!      write(*,*) this_image(), ad1(1,1)[1]
      !$xmp end task
      !$xmp task on q2
!!  allocate(ad2(1:10,1:10)[*])
!!  ad2=this_image()
!!      write(*,*) this_image(), ad2(1,1)[1]
      !$xmp end task
      !$xmp end tasks


  nerr=0

  if (nerr==0) then 
     print '("[",i0,"] OK")', this_image()
  else
     print '("[",i0,"] number of NGs: ",i0)', this_image(), nerr
  end if

  end 


