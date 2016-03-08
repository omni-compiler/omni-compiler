  interface foo
     subroutine foo1(a)
       real a
     end subroutine foo1
     subroutine foo2(a)
       character(*) a
     end subroutine foo2
  end interface

  call foo("abc")
  end
