      ! Tests for the specification expression
      PROGRAM main
      CONTAINS
        SUBROUTINE sub(arg1)
          USE ISO_FORTRAN_ENV
          USE ISO_C_BINDING

          INTEGER :: arg1

          TYPE t
             INTEGER v
          END TYPE t

          INTEGER :: a

          INTEGER,PARAMETER,DIMENSION(2) :: param_array = (/1,2/)
          TYPE(t),PARAMETER :: param_struct = t(1)

          ! constant
          CHARACTER(len=1)                 :: c001       ! a constant integer

          ! subobject of constant

          INTEGER                          :: i001 = param_array(param_array(1))
          CHARACTER(len=param_array(1))    :: c002 
          ! CHARACTER(len=param_struct%v)    :: c003

          ! dummy argument
          CHARACTER(len=arg1)              :: c004


          ! 2008 feature

          !! NEW_LINE intrinsic

          !! *NOT IMPLEMENTED*
          !! type parameter inquery

          ! CHARACTER(len=param_array%len) :: c008

          !! IEEE funciton inquery from the intrinsic IEEE_* module

          ! other?

          !! C_SIZEOF from the ISO_C_BINDING module

          CHARACTER(len=C_SIZEOF(arg1)) :: c_c_sizeof

          !! intrinsic module ISO_FORTRAN_ENV

          !!! COMPLIER VERSION or COMPILER OPTIONS inquiry function from the

          CHARACTER(len=LEN(COMPILER_VERSION())) :: c_compiver_version = COMPILER_VERSION()

          CHARACTER(len=LEN(COMPILER_OPTIONS())) :: c_compiver_options = COMPILER_OPTIONS()


        END SUBROUTINE sub
      END PROGRAM main
