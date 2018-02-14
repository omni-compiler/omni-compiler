program test

  !  integer, parameter :: ii = selected_int_kind(55)
  character :: s = 4_"hello"

  INTEGER, PARAMETER :: ascii = SELECTED_CHAR_KIND('ASCII')

  INTEGER, PARAMETER :: ascii = SELECTED_CHAR_KIND('ASCII')
  INTEGER, PARAMETER :: ucs = SELECTED_CHAR_KIND('ISO_10646')

  PRINT *, ascii_"HELLO WORLD"
  PRINT *, ucs_"こんにちわ世界"
  PRINT *, 1_"HELLO WORLD"
  PRINT *, 4_"こんにちわ世界"

  PRINT *, ascii_"HELLO WORLD"
  k = 6_4
!  kk = 100_ii

END program test
