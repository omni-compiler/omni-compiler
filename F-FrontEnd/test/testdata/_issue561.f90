MODULE issue561_mod

  IMPLICIT NONE

  TYPE, ABSTRACT :: t_var
    REAL, ALLOCATABLE      :: field(:,:,:,:,:)
  END TYPE t_var

  TYPE, EXTENDS(t_var) :: t_real2d
    REAL, POINTER :: ptr(:,:) => NULL()
  CONTAINS
    FINAL :: finalize_real2d
  END TYPE t_real2d

  TYPE, EXTENDS(t_var) :: t_real3d
    REAL, POINTER :: ptr(:,:,:) => NULL()
  CONTAINS
    FINAL :: finalize_real3d
  END TYPE t_real3d

CONTAINS

  SUBROUTINE finalize_real2d(this)
    TYPE(t_real2d) :: this
  END SUBROUTINE finalize_real2d

  SUBROUTINE finalize_real3d(this)
    TYPE(t_real3d) :: this
  END SUBROUTINE finalize_real3d

END MODULE issue561_mod
