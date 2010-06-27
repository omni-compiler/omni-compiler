      SUBROUTINE manipulate_stack
        IMPLICIT NONE
        INTEGER :: value
        INTEGER :: size
        INTEGER :: top = 0
        PARAMETER (size = 100)
        INTEGER, DIMENSION(size) :: stack
        SAVE stack, top

        ENTRY push(value)       ! Push value onto the stack
          IF (top == size) STOP 'Stack Overflow'
          top = top + 1
          stack(top) = value
          RETURN

        ENTRY pop(value)        ! Pop top of stack and place in value
          IF (top == 0) STOP 'Stack Underflow'
          value = stack(top)
          top = top - 1
          RETURN
      END SUBROUTINE manipulate_stack

      PROGRAM main
        integer x1, x2

        call push(10)
        call push(20)

        call pop(x1)
        call pop(x2)

        print *, x1, x2
      END PROGRAM
