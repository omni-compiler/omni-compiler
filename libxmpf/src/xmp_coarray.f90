  module xmp_coarray

    use xmp_coarray_alloc
    use xmp_coarray_get
    use xmp_coarray_sync

    !-------------------------------
    !  coarray intrinsics
    !-------------------------------
    !! inquiry functions
    !     integer, external :: image_index
    !     integer, external :: lcobound, ucobound

    !! transformation functions
    integer, external :: num_images, this_image

    !-------------------------------
    !  coarray runtime interface
    !-------------------------------

    !! others
    integer, external :: xmpf_coarray_image

    !      external :: xmpf_coarray_proc_init
    !      external :: xmpf_coarray_proc_finalize

    !      interface
    !         subroutine xmpf_coarray_proc_init(tag)
    !           integer(8), intent(out):: tag
    !         end subroutine xmpf_coarray_proc_init
    !         subroutine xmpf_coarray_proc_finalize(tag)
    !           integer(8), intent(in):: tag
    !         end subroutine xmpf_coarray_proc_finalize
    !      end interface

  end module xmp_coarray
