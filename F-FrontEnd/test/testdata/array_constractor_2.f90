      program main
        real, dimension(3)   :: vector3d
        character(LEN=10), dimension(2) :: char2d
        vector3d = [ 1.0, 2.0, 3.0 * 2 ]
        char2d = [ "abcef", "DEFGH" ]
        vector3d = [ REAL :: 1.0, 2.0, 3.0 * 2 ]
        char2d = [ CHARACTER(LEN=10) :: "abcef", "DEFGH" ]
      end
