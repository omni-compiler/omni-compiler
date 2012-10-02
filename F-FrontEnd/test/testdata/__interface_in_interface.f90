module mod_interface
  interface a
  subroutine sub(i)
    interface
      subroutine sub1(i)
      integer i
      end subroutine sub1
    end interface
  end subroutine sub
  end interface a
end module mod_interface
