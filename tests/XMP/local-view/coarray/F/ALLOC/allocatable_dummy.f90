call sub(5,7)
call sub(6,8)
end program

  subroutine sub(n1,n2)
    real, allocatable,save:: a(:)
    interface 
       subroutine alloc(a, n1,n2)
         integer n1,n2
         real, allocatable:: a(:)
       end subroutine alloc
    end interface

    if (.not.allocated(a)) then
       call alloc(a, n1,n2)
    endif
    write(*,*) "n1,n2,lb,ub", n1,n2,lbound(a,1), ubound(a,1)

  end subroutine sub

  subroutine alloc(a, n1,n2)
    integer n1,n2
    real, allocatable:: a(:)
    allocate(a(n1:n2))
  end subroutine alloc


    
