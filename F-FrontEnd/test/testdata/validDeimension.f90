      subroutine sub1( validAssumedShapeArray1,   &
                     & validAssumedShapeArray2,   &
                     & validAssumedShapeArray3,   &
                     & validAssumedShapeArray4,   &
                     & validAssumedShapeArray5,   &
                     & validAssumedShapeArray6,   &
                     & validAssumedSizeArray1,    &
                     & validAssumedSizeArray2,    &
                     & validAssumedSizeArray3,    &
                     & validAssumedSizeArray4,    &
                     & inValidAssumedShapeArray1, &
                     & inValidAssumedShapeArray2, &
                     & inValidAssumedShapeArray3, &
                     & inValidAssumedShapeArray4, &
                     & inValidAssumedSizeArray1,  &
                     & inValidAssumedSizeArray2,  &
                     & inValidAssumedSizeArray3   )

        ! Assumed Shape Array.
        character, dimension(:)            :: validAssumedShapeArray1
        character, dimension(:,:)          :: validAssumedShapeArray2
        character, dimension(3:,3:)        :: validAssumedShapeArray3

        character                          :: validAssumedShapeArray4(:)
        character                          :: validAssumedShapeArray5(:,:)
        character                          :: validAssumedShapeArray6(3:,3:)

!        character, dimension(3,:)          :: inValidAssumedShapeArray1
!        character, dimension(:3)           :: inValidAssumedShapeArray2
!        character, dimension(:,*)          :: inValidAssumedShapeArray3
!        character, dimension(*,:)          :: inValidAssumedShapeArray4

        ! Assumed Size Array
        character, dimension(*)            :: validAssumedSizeArray1
        character, dimension(3,5,*)        :: validAssumedSizeArray2

        character                          :: validAssumedSizeArray3(*)
        character                          :: validAssumedSizeArray4(5,*)

!        character, dimension(*5)           :: inValidAssumedSizeArray1
!        character, dimension(5*)           :: inValidAssumedSizeArray2
!        character, dimension(*,5)          :: inValidAssumedSizeArray3

        ! array pointer
        character, pointer, dimension(:)   :: validArrayPointer1
        character, pointer, dimension(:,:) :: validArrayPointer2

        character, pointer                 :: validArrayPointer3(:)
        character, pointer                 :: validArrayPointer4(:,:)

!        character, pointer, dimension(3,:) :: inValidArrayPointer1
!        character, pointer, dimension(:,3) :: inValidArrayPointer2
!        character, pointer, dimension(:3)  :: inValidArrayPointer3
!        character, pointer, dimension(3:)  :: inValidArrayPointer4
!        character, pointer, dimension(1:3) :: inValidArrayPointer5
!        character, pointer, dimension(*)   :: inValidArrayPointer1

        ! allocatable array
        character, allocatable, dimension(:)   :: validAllocatableArray1
        character, allocatable, dimension(:,:) :: validAllocatableArray2

        character, allocatable                 :: validAllocatableArray3(:)
        character, allocatable                 :: validAllocatableArray4(:,:)

!        character, allocatable, dimension(3,:) :: inValidAllocatableArray1
!        character, allocatable, dimension(:,3) :: inValidAllocatableArray2
!        character, allocatable, dimension(:3)  :: inValidAllocatableArray3
!        character, allocatable, dimension(3:)  :: inValidAllocatableArray4
!        character, allocatable, dimension(1:3) :: inValidAllocatableArray5
!        character, allocatable, dimension(*)   :: inValidAllocatableArray1

      end subroutine sub1
