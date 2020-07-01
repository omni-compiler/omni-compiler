subroutine sub2(a)

  INTERFACE
     SUBROUTINE sub1
     END SUBROUTINE sub1
  END INTERFACE

  INTERFACE sub0
     PROCEDURE sub1
  END interface sub0

  real a(10)
  
end subroutine sub2
