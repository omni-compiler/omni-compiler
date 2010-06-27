program main
    integer, external::omp_get_num_threads
    print *, omp_get_num_threads()
    !$omp barrier
    print *, "after barrier"
end program

