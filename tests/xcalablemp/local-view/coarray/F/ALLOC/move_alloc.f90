call sub(5,7)
call sub(6,8)
end program

  subroutine sub(n1,n2)
    real, allocatable,save:: a(:)
    real, allocatable:: b(:)
    interface 
       subroutine alloc(a, n1,n2)
         integer n1,n2
         real, allocatable:: a(:)
       end subroutine alloc
    end interface

    call alloc(a, n1,n2)
    write(*,*) allocated(a), allocated(b)
    call move_alloc(a,b)
    write(*,*) allocated(a), allocated(b)
    write(*,*) "[lu]bound(b,1)=",lbound(b,1),ubound(b,1)

  end subroutine sub

  subroutine alloc(a, n1,n2)
    integer n1,n2
    real, allocatable:: a(:)
    allocate(a(n1:n2))
  end subroutine alloc


    
