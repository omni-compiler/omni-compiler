      module m
        use m_abstract_derived_type

        TYPE, EXTENDS(t) :: tt
         CONTAINS
          PROCEDURE :: p => ftt
        END TYPE
       contains
        FUNCTION ftt(this)
          CLASS(tt) :: this
        END FUNCTION ftt
      end module m

      program main
        use m
      end program main
