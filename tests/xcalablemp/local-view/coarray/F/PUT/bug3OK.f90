program bug3
  interface foo

     subroutine foo2d(x)
       integer x(:,:)
     end subroutine foo2d

     subroutine foo1d(x)
       integer x(:)
     end subroutine foo1d

  end interface

  integer a(10,20)

  call foo(a)
end program
