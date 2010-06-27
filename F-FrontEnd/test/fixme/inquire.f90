program main
   integer :: io_record_length    ! actual record length

   integer , dimension(:), allocatable :: work_i

    inquire (iolength=io_record_length) work_i
end program main
