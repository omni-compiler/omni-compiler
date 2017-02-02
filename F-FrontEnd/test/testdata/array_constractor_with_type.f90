      program main
        type t
          integer :: v
        end type t
        real, dimension(3)   :: vector3d
        character(LEN=10), dimension(2) :: char2d
        type(t), dimension(3)   :: derivedType3d
        vector3d = (/ REAL :: 1.0, 2.0, 3.0 * 2/)
        char2d = (/ CHARACTER(len=10) ::  "abcef", "DEFGH"/)
        char2d = (/ CHARACTER(len=3) ::  "abcef", "DEFGH"/)
        derivedType3d = [ t :: t(1), t(2), t(3) ]
      end
