program bug3
  interface foo

     subroutine foo1d(x)
       integer x(:)
     end subroutine foo1d

     subroutine foo2d(x)
       integer x(:,:)
     end subroutine foo2d

  end interface

  integer a(10)

  call foo(a)
end program
