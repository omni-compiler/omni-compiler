subroutine sub1
  REAL SCORE01(:), NAMES01(:,:)
  ALLOCATABLE:: SCORE01, NAMES01
  integer rec011, rec012
  ALLOCATABLE:: REC1(: ,: , :), REC2(: ,: , :)
end


subroutine sub
  REAL SCORE00(:), NAMES00(:,:)
  ALLOCATABLE SCORE00, NAMES00
  integer rec001, rec002
  ALLOCATABLE REC001(: ,: , :), REC002(: ,: , :)
end

program test
  REAL SCORE(:), NAMES(:,:)
  ALLOCATABLE SCORE
  ALLOCATABLE:: NAMES
  integer rec1, rec2
  ALLOCATABLE:: REC1(: ,: , :)
  ALLOCATABLE REC2(: ,: , :)
end program
