program main
  interface foo
     function foo_i2(n), result(r)
       integer(2) n,r
     end function foo_i2
  
     function foo_i4(n), result(r)
       integer(4) n, r
     end function foo_i4
  end interface

  integer(kind=2) :: n,m
  n = 1_2
  m = foo(n)
  m = foo(-max(n,n))
  m = foo(foo(n))
end program main

function foo_i2(n), result(r)
  integer(2) n, r
  write(*,*) "I am foo_i2"
  r=n
end function foo_i2

function foo_i4(n), result(r)
  integer(4) n, r
  write(*,*) "I am foo_i4"
  r=n
end function foo_i4

