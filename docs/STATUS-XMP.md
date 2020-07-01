Implementation Status version 1.3.1 or later
---------------------------------------
# Global-View
* Mot support block(n) distribution in distribute directive in XMP/C
* Not support master-IO
* Not support global-IO in XMP/C
* Not support tasks directive
* In a task construct, a section of a node array or a template shall not appear in the on clause of a directive.
* In XMP/C for function prototype, a multidimensional array is used in argument of the function,
  the function prototype can not be used.

# Local-View
* Not support "coarray" and "image" directives in the XMP Specification v1.2.1
* Not support "post/wait" and "lock/unlock" directives in XMP/Fortran
* Not support "lock/unlock" directives on the K computer/FX10/FX100
* Not support "lock/unlock" directives when using MPI3

# Intrinsic and Library Procedures
* Not support the following functions
    * xmp_get_primary_image_index()
    * xmp_get_image_index()
    * xmp_sync_image()
    * xmp_sync_images_all()
    * xmp_nodes_attr()

# Known issues
## XMP/C
* In "for statement" with loop directive, a distributed array cannot be used with different distribution manner.
* In "for statement" with loop directive, index in a non-distributed array is not correct.
