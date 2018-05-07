MODULE issue533_mod
CONTAINS
  FUNCTION toCharacter(charArray) RESULT(resultVar)
    CHARACTER, INTENT(IN) :: charArray(:)
    CHARACTER(LEN = :), POINTER :: resultVar
    INTEGER :: i, error, stringSize
    stringSize = SIZE(charArray, 1)
    ALLOCATE(CHARACTER(LEN = stringSize) :: resultVar, STAT = error)
  END FUNCTION toCharacter
END MODULE issue533_mod
