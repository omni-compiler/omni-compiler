      MODULE m
        TYPE t
          INTEGER :: u = 10
         CONTAINS
          PROCEDURE, PASS(a) :: add => add_t
          GENERIC :: OPERATOR(+) => add
        END TYPE t
        TYPE, EXTENDS(t) :: tt
          INTEGER :: v = 10
         CONTAINS
          PROCEDURE, PASS(a) :: add => add_tt
          PROCEDURE, PASS(a) :: add_tt_new
          GENERIC :: OPERATOR(+) => add_tt_new
        END TYPE tt
       CONTAINS
         ELEMENTAL FUNCTION add_t(a, b)
           CLASS(t), INTENT(IN) :: a
           INTEGER, INTENT(IN) :: b
           INTEGER:: add_t
           add_t = a%u + b
         END FUNCTION add_t
         ELEMENTAL FUNCTION add_tt(a, b)
           CLASS(tt), INTENT(IN) :: a
           INTEGER, INTENT(IN) :: b
           INTEGER :: add_tt
           add_tt = a%v + b
         END FUNCTION add_tt
         ELEMENTAL FUNCTION add_tt_new(a, b)
           CLASS(tt), INTENT(IN) :: a
           REAL, INTENT(IN) :: b
           INTEGER :: add_tt_new
           add_tt_new = a%v + b
         END FUNCTION add_tt_new
      END MODULE m
