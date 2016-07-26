  type xmp_desc
     sequence
     integer*8 :: desc
  end type xmp_desc

  integer, parameter :: XMP_ENTIRE_NODES      = 2000
  integer, parameter :: XMP_EXECUTING_NODES   = 2001
  integer, parameter :: XMP_PRIMARY_NODES     = 2002
  integer, parameter :: XMP_EQUIVALENCE_NODES = 2003

  integer, parameter :: XMP_NOT_DISTRIBUTED   = 2100
  integer, parameter :: XMP_BLOCK             = 2101
  integer, parameter :: XMP_CYCLIC            = 2102
  integer, parameter :: XMP_GBLOCK            = 2103

  integer, parameter :: XMP_DESC_NODES        = 2200
  integer, parameter :: XMP_DESC_TEMPLATE     = 2201
  integer, parameter :: XMP_DESC_ARRAY        = 2202
