program epi1
  real, allocatable :: a(:)[:] 

  interface
     subroutine allo(a)
       real, allocatable :: a(:)[:] 
       real, allocatable :: al(:)[:] 
     end subroutine allo
  end interface

  write(*,*) "F ? ", allocated(a)
  call allo
  write(*,*) "T ? ", allocated(a)
  call allo
  write(*,*) "T ? ", allocated(a)

end program epi1


subroutine allo(a)
  real, allocatable :: a(:)[:] 
  real, allocatable :: al(:)[:] 
    
  write(*,*) "F ? ", allocated(al)
  allocate (a(10)[*],al(10)[*])
  write(*,*) "T T ? ", allocated(a), allocated(al)

end subroutine allo


