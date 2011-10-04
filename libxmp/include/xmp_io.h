/*==================================================================*/
/* xmp_io.h                                                         */
/* Copyright (C) 2011, FUJITSU LIMITED                              */
/*==================================================================*\
  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation;
  version 2.1 published by the Free Software Foundation.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
\*==================================================================*/

#ifndef _XMP_IO
#define _XMP_IO

// --------------- including headers  --------------------------------
#include "mpi.h"
//#include "xmp_internal.h"

// --------------- structures ----------------------------------------
typedef struct xmp_file_t {
    MPI_File   fh;
    MPI_Offset disp;
    char       is_append;
} xmp_file_t;

typedef struct xmp_range_t /* 入出力の範囲を示す構造体 */
{
  int dims;                /* 分割配列の次元数     */
  int *lb;                 /* 入出力の下限(dims分) */
  int *ub;                 /* 入出力の上限(dims分) */
  int *step;               /* 入出力の増分(dims分) */
} xmp_range_t;

typedef void* xmp_array_t;

// --------------- functions -----------------------------------------
// xmp_io.c
extern xmp_file_t *xmp_fopen_all(const char*, const char*);
extern int        xmp_fclose_all(xmp_file_t*);
extern int        xmp_fseek(xmp_file_t*, long long, int);
extern int        xmp_fseek_shared_all(xmp_file_t*, long long, int);
extern long long  xmp_ftell(xmp_file_t*);
extern long long  xmp_ftell_shared(xmp_file_t*);
extern long long  xmp_file_sync_all(xmp_file_t*);
extern size_t     xmp_fread_all(xmp_file_t*, void*, size_t, size_t);
extern size_t     xmp_fread_darray_all(xmp_file_t*, xmp_array_t, xmp_range_t*);
extern size_t     xmp_fwrite_darray_all(xmp_file_t*, xmp_array_t, xmp_range_t*);
extern size_t     xmp_fwrite_all(xmp_file_t*, void*, size_t, size_t);
extern size_t     xmp_fread_shared(xmp_file_t*, void*, size_t, size_t);
extern size_t     xmp_fwrite_shared(xmp_file_t*, void*, size_t, size_t);
extern size_t     xmp_fread(xmp_file_t*, void*, size_t, size_t);
extern size_t     xmp_fwrite(xmp_file_t*, void*, size_t, size_t);
extern int        xmp_file_set_view_all(xmp_file_t*, long long, xmp_array_t, xmp_range_t*);
extern int        xmp_file_clear_view_all(xmp_file_t*, long long);

#endif // _XMP_IO
