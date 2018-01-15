subroutine foo
  interface
     subroutine bar(b)
       real b(:)
     end subroutine bar
  end interface
  real a(100)
  call bar(a(:))
end subroutine foo
