SUBROUTINE sub (fourier)
  REAL fourier
  EXTERNAL fourier              ! statement form
  REAL, EXTERNAL :: SIN, COS, TAN  ! attribute form
END SUBROUTINE sub

SUBROUTINE gratx (x, y)
  EXTERNAL init_block_a
  COMMON /a/ temp, pressure
END SUBROUTINE gratx
