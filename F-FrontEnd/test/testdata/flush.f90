      program main
        integer data(10)
        integer :: iostat
        character(10) :: iomsg
        data = (/(i,i=1,10)/)
        open(50, file="data_file")
        write(50, *) data
        flush 50
        flush(50)
        flush(unit=50)
        flush(unit=50, iostat=iostat)
        flush(unit=50, iostat=iostat, iomsg=iomsg)
        flush(unit=50, iostat=iostat, iomsg=iomsg, err=111)
        call exit()
  111   write(*,*) "some error"
      end program main
