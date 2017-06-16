      program main
        integer data(10)
        data = (/(i,i=1,10)/)
        open(50, file="data_file")
        write(50, *) data
        flush(50)
      end program main
