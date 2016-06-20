!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
!!
!!  allnodes (IOUNIT_SIZE * N_IOUNIT)
!!
!!        x        x        x        x     ---- ionodes
!!        o        o        o        o
!!        o        o        o        o
!!     iounit   iounit   iounit   iounit 
!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

program test_subcomm_write
  integer,parameter:: IOUNIT_SIZE = 3
  integer,parameter:: N_IOUNIT = 4
  integer,parameter:: IO_NODE_ID = 1

  !$xmp nodes allnodes(IOUNIT_SIZE, N_IOUNIT)
  !$xmp nodes iounit(IOUNIT_SIZE) = allnodes(:,*)
  !$xmp nodes ionodes(N_IOUNIT) = allnodes(IO_NODE_ID,:)

  integer mydata
  integer teamdata(IOUNIT_SIZE)[*]
  !$xmp coarray on iounit:: teamdata

  !!--------------------------------- init
  mydata = this_image()*10
  sync all

  !!--------------------------------- gather mydata->teamdata
  !$xmp task on iounit
  me = this_image()       !! image index in iounit (1<= <=IOUNIT_SIZE)
  teamdata(me)[IO_NODE_ID] = mydata
  !$xmp end task
  sync all

  !!--------------------------------- output teamdata
  !$xmp task on ionodes
  write(*,100) this_image(), teamdata
  !$xmp end task
     
100 format("IO node #",i0," writes: ",i5,i5,i5,i5,i5,i5,i5,i5)

end program test_subcomm_write

