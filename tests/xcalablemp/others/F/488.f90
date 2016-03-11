  subroutine sub
    interface foo
       subroutine foo1(a)
         real a
       end subroutine foo1
       subroutine foo2(n)
         integer n
       end subroutine foo2
    end interface

    continue
  end subroutine sub
