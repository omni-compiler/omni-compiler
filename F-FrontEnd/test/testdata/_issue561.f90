MODULE issue561_mod

  IMPLICIT NONE

  TYPE, ABSTRACT :: t_var
    REAL, ALLOCATABLE      :: field(:,:,:,:,:)
  END TYPE t_var

  TYPE, EXTENDS(t_var) :: t_real2d
    REAL, POINTER :: ptr(:,:) => NULL()
  CONTAINS
    PROCEDURE, PASS :: test => test1
    FINAL :: finalize_real2d
  END TYPE t_real2d

  TYPE, EXTENDS(t_var) :: t_real3d
    REAL, POINTER :: ptr(:,:,:) => NULL()
  CONTAINS
    FINAL :: finalize_real3d
  END TYPE t_real3d

CONTAINS

  SUBROUTINE test1(this)
    CLASS(t_real2d) :: this
  END SUBROUTINE test1

  SUBROUTINE finalize_real2d(this)
    TYPE(t_real2d) :: this
  END SUBROUTINE finalize_real2d

  SUBROUTINE finalize_real3d(this)
    TYPE(t_real3d) :: this
  END SUBROUTINE finalize_real3d

END MODULE issue561_mod
