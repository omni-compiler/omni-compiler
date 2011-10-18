/*==================================================================*/
/* xmp_io.c                                                         */
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

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "xmp_constant.h"
#include "xmp_data_struct.h"
#include "xmp_io.h"

//#define DEBUG

/*****************************************************************************/
/*  FUNCTION NAME : xmp_fopen_all                                            */
/*  DESCRIPTION   : この関数はXcalableMP用のファイルを開く。                 */
/*                  この関数は集団実行しなければならない。                   */
/*  ARGUMENT      : fname[IN] ファイル名                                     */
/*                  amode[IN] POSIXのfopen相当。"rwa+"の組合せ。             */
/*  RETURN VALUES : xmp_file_t* ファイル構造体。異常終了の場合はNULLを返す。 */
/*                                                                           */
/*****************************************************************************/
xmp_file_t *xmp_fopen_all(const char *fname, const char *amode)
{
  xmp_file_t *pstXmp_file = NULL;
  int         iMode = 0;
  size_t      modelen = 0;


  // 領域確保
  pstXmp_file = malloc(sizeof(xmp_file_t));
  if (pstXmp_file == NULL) { return NULL; } 
  memset(pstXmp_file, 0x00, sizeof(xmp_file_t));
  
  ///
  /// モード解析
  ///
  modelen = strlen(amode);
  // モードが１文字
  if (modelen == 1)
  {
    if (strncmp(amode, "r", modelen) == 0)
    {
      iMode = MPI_MODE_RDONLY;
    }
    else if (strncmp(amode, "w", modelen) == 0)
    {
      iMode = (MPI_MODE_WRONLY | MPI_MODE_CREATE);
    }
    else if (strncmp(amode, "a", modelen) == 0)
    {
      iMode = (MPI_MODE_RDWR | MPI_MODE_CREATE | MPI_MODE_APPEND);
      pstXmp_file->is_append = 0x01;
    }
    else
    {
      goto ErrorExit;
    }
  }
  // モード２文字
  else if (modelen == 2)
  {
    if (strncmp(amode, "r+", modelen) == 0)
    {
      iMode = MPI_MODE_RDWR;
    }
    else if (strncmp(amode, "w+", modelen) == 0)
    {
      iMode = (MPI_MODE_RDWR | MPI_MODE_CREATE);
    }
    else if (strncmp(amode, "a+", modelen) == 0 ||
             strncmp(amode, "ra", modelen) == 0 ||
             strncmp(amode, "ar", modelen) == 0)
    {
      iMode = (MPI_MODE_RDWR | MPI_MODE_CREATE);
      pstXmp_file->is_append = 0x01;
    }
    else if (strncmp(amode, "rw", modelen) == 0 ||
             strncmp(amode, "wr", modelen) == 0)
    {
        goto ErrorExit;
    }
    else
    {
      goto ErrorExit;
    }
  }
  // モードその他
  else
  {
    goto ErrorExit;
  }

  // ファイルオープン
  if (MPI_File_open(MPI_COMM_WORLD,
                    (char*)fname,
                    iMode,
                    MPI_INFO_NULL,
                    &(pstXmp_file->fh)) != MPI_SUCCESS)
  {
    goto ErrorExit;
  }

  // "W" or "W+"の場合はファイルサイズを０にする
  if ((iMode == (MPI_MODE_WRONLY | MPI_MODE_CREATE)  ||
       iMode == (MPI_MODE_RDWR   | MPI_MODE_CREATE)) &&
       pstXmp_file->is_append == 0x00)
  {
    if (MPI_File_set_size(pstXmp_file->fh, 0) != MPI_SUCCESS)
    {
      goto ErrorExit;
    }
  }

  // 正常終了
  return pstXmp_file;

// エラー時
ErrorExit:
  if (pstXmp_file != NULL)
  {
    free(pstXmp_file);
  }
  return NULL;
}

/*****************************************************************************/
/*  FUNCTION NAME : xmp_fclose_all                                           */
/*  DESCRIPTION   : この関数はXcalableMP用のファイルを閉じる。               */
/*                  この関数は集団実行しなければならない。                   */
/*  ARGUMENT      : pstXmp_file[IN] ファイル構造体                           */
/*  RETURN VALUES : 正常終了の場合は0を返す。                                */
/*                  異常終了の場合は0以外の値を返す。                        */
/*                                                                           */
/*****************************************************************************/
int xmp_fclose_all(xmp_file_t *pstXmp_file)
{
  // 引数チェック
  if (pstXmp_file == NULL)     { return 1; }

  // ファイルクローズ
  if (MPI_File_close(&(pstXmp_file->fh)) != MPI_SUCCESS)
  {
    free(pstXmp_file);
    return 1;
  }

  free(pstXmp_file);
  return 0;
}


/*****************************************************************************/
/*  FUNCTION NAME : xmp_fseek                                                */
/*  DESCRIPTION   : この関数は、fhの固有ファイルポインタの位置を変更する。   */
/*                  この関数はローカル実行可能である。                       */
/*  ARGUMENT      : pstXmp_file[IN] ファイル構造体                           */
/*                  offset[IN] whenceの位置から現在のファイルビューの変位    */
/*                  whence[IN] ファイルの位置を選択                          */
/*  RETURN VALUES : 正常終了の場合は0を返す。                                */
/*                  異常終了の場合は0以外の値を返す。                        */
/*                                                                           */
/*****************************************************************************/
int xmp_fseek(xmp_file_t *pstXmp_file, long long offset, int whence)
{
  int iMpiWhence;

  // 引数チェック
  if (pstXmp_file == NULL) { return 1; }

  // offsetをMPI_Offsetへ変換
  switch (whence)
  {
    case SEEK_SET:
      iMpiWhence = MPI_SEEK_SET;
      break;
    case SEEK_CUR:
      iMpiWhence = MPI_SEEK_CUR;
      break;
    case SEEK_END:
      iMpiWhence = MPI_SEEK_END;
      break;
    default:
      return 1;
  }

  // ファイルシーク
  if (MPI_File_seek(pstXmp_file->fh, (MPI_Offset)offset, iMpiWhence) != MPI_SUCCESS)
  {
    return 1;
  }

  return 0;
}

/*****************************************************************************/
/*  FUNCTION NAME : xmp_fseek_shared_all                                     */
/*  DESCRIPTION   : この関数は、fhの共有ファイルポインタの位置を変更する。   */
/*                  この関数は集団実行しなければならない。                   */
/*  ARGUMENT      : pstXmp_file[IN] ファイル構造体                           */
/*                  offset[IN] whenceの位置から現在のファイルビューの変位    */
/*                  whence[IN] ファイルの位置を選択                          */
/*  RETURN VALUES : 正常終了の場合は0を返す。                                */
/*                  異常終了の場合は0以外の値を返す。                        */
/*                                                                           */
/*****************************************************************************/
int xmp_fseek_shared(xmp_file_t *pstXmp_file, long long offset, int whence)
{
  int iMpiWhence;

  // 引数チェック
  if (pstXmp_file == NULL) { return 1; }

  // offsetをMPI_Offsetへ変換
  switch (whence)
  {
    case SEEK_SET:
      iMpiWhence = MPI_SEEK_SET;
      break;
    case SEEK_CUR:
      iMpiWhence = MPI_SEEK_CUR;
      break;
    case SEEK_END:
      iMpiWhence = MPI_SEEK_END;
      break;
    default:
      return 1;
  }

  // ファイルシーク
  if (MPI_File_seek_shared(pstXmp_file->fh, (MPI_Offset)offset, iMpiWhence) != MPI_SUCCESS)
  {
    return 1;
  }

  return 0;
}

/*****************************************************************************/
/*  FUNCTION NAME : xmp_ftell                                                */
/*  DESCRIPTION   : この関数は、fhの固有ファイルポインタのファイル先頭からの */
/*                  変位を求める。                                           */
/*                  この関数はローカル実行可能である。                       */
/*  ARGUMENT      : pstXmp_file[IN] ファイル構造体                           */
/*  RETURN VALUES : 正常終了の場合はファイル先頭からの変位(byte)を返す。     */
/*                  異常終了の場合は負数を返す。                             */
/*                                                                           */
/*****************************************************************************/
long long xmp_ftell(xmp_file_t *pstXmp_file)
{
  MPI_Offset offset;
  MPI_Offset disp;

  // 引数チェック
  if (pstXmp_file == NULL) { return -1; }

  if (MPI_File_get_position(pstXmp_file->fh, &offset) != MPI_SUCCESS)
  {
    return -1;
  }

  if (MPI_File_get_byte_offset(pstXmp_file->fh, offset, &disp) != MPI_SUCCESS)
  {
    return -1;
  }

  return (long long)disp;
}

/*****************************************************************************/
/*  FUNCTION NAME : xmp_ftell_shared                                         */
/*  DESCRIPTION   : この関数は、fhの共有ファイルポインタのファイル先頭からの */
/*                  変位を求める。                                           */
/*                  この関数はローカル実行可能である。                       */
/*  ARGUMENT      : pstXmp_file[IN] ファイル構造体                           */
/*  RETURN VALUES : 正常終了の場合はファイル先頭からの変位(byte)を返す。     */
/*                  異常終了の場合は負数を返す。                             */
/*                                                                           */
/*****************************************************************************/
long long xmp_ftell_shared(xmp_file_t *pstXmp_file)
{
  MPI_Offset offset;
  MPI_Offset disp;

  // 引数チェック
  if (pstXmp_file == NULL) { return -1; }

  if (MPI_File_get_position_shared(pstXmp_file->fh, &offset) != MPI_SUCCESS)
  {
    return -1;
  }

  if (MPI_File_get_byte_offset(pstXmp_file->fh, offset, &disp) != MPI_SUCCESS)
  {
    return -1;
  }

  return (long long)disp;
}

/*****************************************************************************/
/*  FUNCTION NAME : xmp_file_sync_all                                        */
/*  DESCRIPTION   : この関数は、ファイルの同期を行う。                       */
/*                  この関数は集団実行しなければならない。                   */
/*  ARGUMENT      : pstXmp_file[IN] ファイル構造体                           */
/*  RETURN VALUES : 正常終了の場合は0を返す。                                */
/*                  異常終了の場合は0以外の値を返す。                        */
/*                                                                           */
/*****************************************************************************/
long long xmp_file_sync_all(xmp_file_t *pstXmp_file)
{
  // 引数チェック
  if (pstXmp_file == NULL) { return 1; }

  // 同期 
  if (MPI_File_sync(pstXmp_file->fh) != MPI_SUCCESS)
  {
    return 1;
  }

  // バリア
  MPI_Barrier(MPI_COMM_WORLD);

  return 0;
}

/*****************************************************************************/
/*  FUNCTION NAME : xmp_fread_all                                            */
/*  DESCRIPTION   : この関数は実行したノードのbufferへファイルビューに従い   */
/*                  データを読込む。                                         */
/*                  この関数は集団実行しなければならない。                   */
/*  ARGUMENT      : pstXmp_file[IN] ファイル構造体                           */
/*                  buffer[OUT] データを読込む変数の先頭アドレス             */
/*                  size[IN]  読込むデータの1要素当りのサイズ (バイト)       */
/*                  count[IN] 読込むデータの数                               */
/*  RETURN VALUES : 正常終了の場合は読込んだバイト数を返す。                 */
/*                  異常終了の場合は負数を返す。                             */
/*                                                                           */
/*****************************************************************************/
size_t xmp_fread_all(xmp_file_t *pstXmp_file, void *buffer, size_t size, size_t count)
{
  MPI_Status status;
  int readCount;

  // 引数チェック
  if (pstXmp_file == NULL) { return -1; }
  if (buffer      == NULL) { return -1; }
  if (size  < 1) { return -1; }
  if (count < 1) { return -1; }

  // 読込み
  if (MPI_File_read_all(pstXmp_file->fh,
                        buffer, size * count,
                        MPI_BYTE,
                        &status) != MPI_SUCCESS)
  {
    return -1;
  }
  
  // 読込んだバイト数
  if (MPI_Get_count(&status, MPI_BYTE, &readCount) != MPI_SUCCESS)
  {
    return -1;
  }

  return readCount;
}

/*****************************************************************************/
/*  FUNCTION NAME : xmp_fwrite_all                                           */
/*  DESCRIPTION   : この関数は実行したノードのbufferからファイルビューに     */
/*                  従いデータを書込む。                                     */
/*                  この関数は集団実行しなければならない。                   */
/*  ARGUMENT      : pstXmp_file[IN] ファイル構造体                           */
/*                  buffer[IN] データを書込む変数の先頭アドレス              */
/*                  size[IN] 書込むデータの1要素当りのサイズ (バイト)        */
/*                  count[IN] 書込むデータの数                               */
/*  RETURN VALUES : 正常終了の場合は書込んだバイト数を返す。                 */
/*                  異常終了の場合は負数を返す。                             */
/*                                                                           */
/*****************************************************************************/
size_t xmp_fwrite_all(xmp_file_t *pstXmp_file, void *buffer, size_t size, size_t count)
{
  MPI_Status status;
  int writeCount;

  // 引数チェック
  if (pstXmp_file == NULL) { return -1; }
  if (buffer      == NULL) { return -1; }
  if (size  < 1) { return -1; }
  if (count < 1) { return -1; }

  // ファイルオープンが"r+"の場合は終端にポインタを移動
  if (pstXmp_file->is_append)
  {
    if (MPI_File_seek(pstXmp_file->fh,
                      (MPI_Offset)0,
                      MPI_SEEK_END) != MPI_SUCCESS)
    {
      return -1;
    }

    pstXmp_file->is_append = 0x00;
  }

  // 書込み
  if (MPI_File_write_all(pstXmp_file->fh, buffer, size * count, MPI_BYTE, &status) != MPI_SUCCESS)
  {
    return -1;
  }

  // 書込んだバイト数
  if (MPI_Get_count(&status, MPI_BYTE, &writeCount) != MPI_SUCCESS)
  {
    return -1;
  }

  return writeCount;
}

/*****************************************************************************/
/*  FUNCTION NAME : xmp_fread_darray_unpack                                  */
/*  DESCRIPTION   : この関数はapで指定される分散配列について、rpで指定される */
/*                  範囲へファイルからデータを読込む。                       */
/*                  派生型は生成せずにアンパックする。                       */
/*                  この関数は集団実行しなければならない。                   */
/*  ARGUMENT      : fp[IN] ファイル構造体                                    */
/*                  ap[IN/OUT] 分散配列情報                                  */
/*                  rp[IN]     アクセス範囲情報                              */
/*  RETURN VALUES : 正常終了の場合は読込んだバイト数を返す。                 */
/*                  異常終了の場合は負数を返す。                             */
/*                                                                           */
/*****************************************************************************/
int xmp_fread_darray_unpack(fp, ap, rp)
     xmp_file_t *fp;
     xmp_array_t ap;
     xmp_range_t *rp;
{
   MPI_Status    status;
   _XMP_array_t *array_t;
   char         *array_addr;
   char         *buf=NULL;
   char         *cp;
   int          *lb=NULL;
   int          *ub=NULL;
   int          *step=NULL;
   int          *cnt=NULL;
   int           buf_size;
   int           ret=0;
   int           disp;
   int           size;
   int           array_size;
   int           i, j;

   array_t = (_XMP_array_t*)ap;
  
   /* 回転数を示す配列確保 */
   lb = (int*)malloc(sizeof(int)*rp->dims);
   ub = (int*)malloc(sizeof(int)*rp->dims);
   step = (int*)malloc(sizeof(int)*rp->dims);
   cnt = (int*)malloc(sizeof(int)*rp->dims);
   if(!lb || !ub || !step || !cnt){
      ret = -1;
      goto FunctionExit;
   }
  
   /* 回転数を求める */
   buf_size = 1;
   for(i=0; i<rp->dims; i++){
      /* error check */
      if(rp->step[i] > 0 && rp->lb[i] > rp->ub[i]){
         ret = -1;
         goto FunctionExit;
      }
      if(rp->step[i] < 0 && rp->lb[i] < rp->ub[i]){
         ret = -1;
         goto FunctionExit;
      }
      if (array_t->info[i].align_manner == _XMP_N_ALIGN_NOT_ALIGNED ||
          array_t->info[i].align_manner == _XMP_N_ALIGN_DUPLICATION) {
         lb[i] = rp->lb[i];
         ub[i] = rp->ub[i];
         step[i] = rp->step[i];
  
      } else if(array_t->info[i].align_manner == _XMP_N_ALIGN_BLOCK){
         if(rp->step[i] > 0){
            if(array_t->info[i].par_upper < rp->lb[i] ||
               array_t->info[i].par_lower > rp->ub[i]){
               lb[i] = 1;
               ub[i] = 0;
               step[i] = 1;
            } else {
               lb[i] = (array_t->info[i].par_lower > rp->lb[i])?
                  rp->lb[i]+((array_t->info[i].par_lower-1-rp->lb[i])/rp->step[i]+1)*rp->step[i]:
                  rp->lb[i];
               ub[i] = (array_t->info[i].par_upper < rp->ub[i]) ?
                  array_t->info[i].par_upper:
                  rp->ub[i];
               step[i] = rp->step[i];
            }
         } else {
            if(array_t->info[i].par_upper < rp->ub[i] ||
               array_t->info[i].par_lower > rp->lb[i]){
               lb[i] = 1;
               ub[i] = 0;
               step[i] = 1;
            } else {
               lb[i] = (array_t->info[i].par_upper < rp->lb[i])?
                  rp->lb[i]-((rp->lb[i]-array_t->info[i].par_upper-1)/rp->step[i]-1)*rp->step[i]:
                  rp->lb[i];
               ub[i] = (array_t->info[i].par_lower > rp->ub[i])?
                  array_t->info[i].par_lower:
                  rp->ub[i];
               step[i] = rp->step[i];
            }
         }
      } else {
         ret = -1;
         goto FunctionExit;
      }
      cnt[i] = (ub[i]-lb[i]+step[i])/step[i];
      cnt[i] = (cnt[i]>0)? cnt[i]: 0;
      buf_size *= cnt[i];
   }
  
   /* バッファ確保 */
   if(buf_size == 0){
      buf = (char*)malloc(array_t->type_size);
   } else {
      buf = (char*)malloc(buf_size*array_t->type_size);
   }
   if(!buf){
      ret = -1;
      goto FunctionExit;
   }

   // 書込み
   if(buf_size > 0){
      if (MPI_File_read(fp->fh, buf, buf_size*array_t->type_size, MPI_BYTE, &status) != MPI_SUCCESS) {
         ret = -1;
         goto FunctionExit;
      }
      
      // 読み込んだバイト数
      if (MPI_Get_count(&status, MPI_BYTE, &ret) != MPI_SUCCESS) {
         ret = -1;
         goto FunctionExit;
      }
   } else {
      ret = 0;
   }
  
   /* データをアンパック */
   cp = buf;
   array_addr = (char*)(*array_t->array_addr_p);
   for(j=0; j<buf_size; j++){
      disp = 0;
      size = 1;
      array_size = 1;
      for(i=rp->dims-1; i>=0; i--){
         ub[i] = (j/size)%cnt[i];
         if (array_t->info[i].align_manner == _XMP_N_ALIGN_NOT_ALIGNED ||
             array_t->info[i].align_manner == _XMP_N_ALIGN_DUPLICATION) {
            disp += (lb[i]+ub[i]*step[i])*array_size;
            array_size *= array_t->info[i].ser_size;
         } else if(array_t->info[i].align_manner == _XMP_N_ALIGN_BLOCK){
            disp += (lb[i]+ub[i]*step[i]+array_t->info[i].local_lower-array_t->info[i].par_lower)*array_size;
            array_size *= array_t->info[i].alloc_size;
         }
         size *= cnt[i];
      }
      disp *= array_t->type_size;
      memcpy(array_addr+disp, cp, array_t->type_size);
      cp += array_t->type_size;
   }

 FunctionExit:
   if(buf) free(buf);
   if(lb) free(lb);
   if(ub) free(ub);
   if(step) free(step);
   if(cnt) free(cnt);

   return ret;
}

/*****************************************************************************/
/*  FUNCTION NAME : xmp_fread_darray_all                                     */
/*  DESCRIPTION   : この関数はapで指定される分散配列について、rpで指定される */
/*                  範囲へファイルからデータを読込む。                       */
/*                  この関数は集団実行しなければならない。                   */
/*  ARGUMENT      : pstXmp_file[IN] ファイル構造体                           */
/*                  ap[IN/OUT] 分散配列情報                                  */
/*                  rp[IN]     アクセス範囲情報                              */
/*  RETURN VALUES : 正常終了の場合は読込んだバイト数を返す。                 */
/*                  異常終了の場合は負数を返す。                             */
/*                                                                           */
/*****************************************************************************/
size_t xmp_fread_darray_all(xmp_file_t  *pstXmp_file,
                            xmp_array_t  ap,
                            xmp_range_t *rp)
{
  _XMP_array_t *XMP_array_t;
  MPI_Status status;        // MPIステータス
  int readCount;            // リードバイト
  int mpiRet;               // MPI関数戻り値
  int lower;                // このノードでアクセスする下限
  int upper;                // このノードでアクセスする上限
  int continuous_size;      // 連続域サイズ
  int space_size;           // 隙間サイズ
  int total_size;           // 全体サイズ
  int type_size;
  MPI_Aint tmp1, tmp2;
  MPI_Datatype dataType[2];
  int i = 0;
#ifdef DEBUG
  int rank, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
#endif

  // 引数チェック
  if (pstXmp_file == NULL) { return -1; }
  if (ap == NULL)          { return -1; }
  if (rp == NULL)          { return -1; }

  /* アンパックが必要な場合は別処理 */
  for (i = rp->dims - 1; i >= 0; i--){
     if(rp->step[i] < 0){
        int ret = xmp_fread_darray_unpack(pstXmp_file, ap, rp);
        return ret;
     }
  }

  XMP_array_t = (_XMP_array_t*)ap; 

  // 次元数のチェック
  if (XMP_array_t->dim != rp->dims) { return -1; }
#ifdef DEBUG
printf("READ(%d/%d) dims=%d\n", rank, nproc, rp->dims);
#endif

  // 基本データ型の作成
  MPI_Type_contiguous(XMP_array_t->type_size, MPI_BYTE, &dataType[0]);

  // 次元数分ループ
  for (i = rp->dims - 1; i >= 0; i--)
  {
#ifdef DEBUG
printf("READ(%d/%d) (lb,ub,step)=(%d,%d,%d)\n",
       rank, nproc, rp->lb[i],  rp->ub[i], rp->step[i]);
printf("READ(%d/%d) (par_lower,par_upper)=(%d,%d)\n",
       rank, nproc, XMP_array_t->info[i].par_lower, XMP_array_t->info[i].par_upper);
#endif
    // 分散の無い次元
    if (XMP_array_t->info[i].align_manner == _XMP_N_ALIGN_NOT_ALIGNED ||
        XMP_array_t->info[i].align_manner == _XMP_N_ALIGN_DUPLICATION)
    {
      // 分割後上限 < 下限
      if (XMP_array_t->info[i].par_upper < rp->lb[i]) { return -1; }
      // 分割後下限 > 上限
      if (XMP_array_t->info[i].par_lower > rp->ub[i]) { return -1; }

      // 増分が負
      if ( rp->step[i] < 0)
      {
      }
      // 増分が正
      else
      {
        // 連続域のサイズ
        continuous_size = (rp->ub[i] - rp->lb[i]) / rp->step[i] + 1;

        // データ型の範囲を取得
        mpiRet =MPI_Type_get_extent(dataType[0], &tmp1, &tmp2);
        if (mpiRet !=  MPI_SUCCESS) { return -1; }  
        type_size = (int)tmp2;

        // 基本データ型の生成
        mpiRet = MPI_Type_create_hvector(continuous_size,
                                         1,
                                         type_size * rp->step[i],
                                         dataType[0],
                                         &dataType[1]);

        // 使用しなくなったMPI_Datatypeを解放
        MPI_Type_free(&dataType[0]);

        // MPI_Type_create_hvectorがエラーの場合
        if (mpiRet != MPI_SUCCESS) { return -1; }

        // 全体サイズ
        total_size
          = (XMP_array_t->info[i].par_upper
          -  XMP_array_t->info[i].par_lower + 1)
          *  type_size;

        // 隙間サイズ
        space_size
          = (rp->lb[i] - XMP_array_t->info[i].par_lower)
          * type_size;

        // 新しいファイル型の作成
        mpiRet = MPI_Type_create_resized(dataType[1],
                                         (MPI_Aint)space_size,
                                         (MPI_Aint)total_size,
                                         &dataType[0]);

        // MPI_Type_create_resizedがエラーの場合
        if (mpiRet != MPI_SUCCESS) { return -1; }

        // 使用しなくなったMPI_Datatypeを解放
        MPI_Type_free(&dataType[1]);

#ifdef DEBUG
printf("READ(%d/%d) NOT_ALIGNED\n", rank, nproc);
printf("READ(%d/%d) continuous_size=%d\n", rank, nproc, continuous_size);
printf("READ(%d/%d) space_size=%d\n", rank, nproc, space_size);
printf("READ(%d/%d) total_size=%d\n", rank, nproc, total_size);
#endif
      }
    }
     // block分散
    else if (XMP_array_t->info[i].align_manner == _XMP_N_ALIGN_BLOCK)
    {
      // 増分が負
      if ( rp->step[i] < 0)
      {
      }
      // 増分が正
      else
      {
        // 分割後上限 < 下限
        if (XMP_array_t->info[i].par_upper < rp->lb[i])
        {
          continuous_size = 0;
        }
        // 分割後下限 > 上限
        else if (XMP_array_t->info[i].par_lower > rp->ub[i])
        {
          continuous_size = 0;
        }
        // その他
        else
        {
          // ノードの下限
          lower
            = (XMP_array_t->info[i].par_lower > rp->lb[i]) ?
              rp->lb[i] + ((XMP_array_t->info[i].par_lower - 1 - rp->lb[i]) / rp->step[i] + 1) * rp->step[i]
            : rp->lb[i];

          // ノードの上限
          upper
            = (XMP_array_t->info[i].par_upper < rp->ub[i]) ?
               XMP_array_t->info[i].par_upper : rp->ub[i];

          // 連続要素数
          continuous_size = (upper - lower + rp->step[i]) / rp->step[i];
        }

        // データ型の範囲を取得
        mpiRet =MPI_Type_get_extent(dataType[0], &tmp1, &tmp2);
        if (mpiRet !=  MPI_SUCCESS) { return -1; }  
        type_size = (int)tmp2;

        // 基本データ型の生成
        mpiRet = MPI_Type_create_hvector(continuous_size,
                                         1,
                                         type_size * rp->step[i],
                                         dataType[0],
                                         &dataType[1]);

        // 使用しなくなったMPI_Datatypeを解放
        MPI_Type_free(&dataType[0]);

        // MPI_Type_create_hvectorがエラーの場合
        if (mpiRet != MPI_SUCCESS) { return -1; }

        // 隙間サイズ
        space_size
          = (XMP_array_t->info[i].local_lower
          + (lower - XMP_array_t->info[i].par_lower))
          * type_size;

        // 全体サイズ
        total_size = (XMP_array_t->info[i].alloc_size)* type_size;

        // 新しいファイル型の作成
        mpiRet = MPI_Type_create_resized(dataType[1],
                                         (MPI_Aint)space_size,
                                         (MPI_Aint)total_size,
                                         &dataType[0]);

        // MPI_Type_create_resizedがエラーの場合
        if (mpiRet != MPI_SUCCESS) { return -1; }

        // 使用しなくなったMPI_Datatypeを解放
        MPI_Type_free(&dataType[1]);

#ifdef DEBUG
printf("READ(%d/%d) ALIGN_BLOCK\n", rank, nproc);
printf("READ(%d/%d) continuous_size=%d\n", rank, nproc, continuous_size);
printf("READ(%d/%d) space_size=%d\n", rank, nproc, space_size);
printf("READ(%d/%d) total_size=%d\n", rank, nproc, total_size);
printf("READ(%d/%d) (lower,upper)=(%d,%d)\n", rank, nproc, lower, upper);
#endif
      }
    }
    // cyclic分散
    else if (XMP_array_t->info[i].align_manner == _XMP_N_ALIGN_CYCLIC)
    {
      return -1;
    }
    // その他
    else
    {
      return -1;
    }
  }

  // コミット
  mpiRet = MPI_Type_commit(&dataType[0]);

  // コミットがエラーの場合
  if (mpiRet != MPI_SUCCESS) { return 1; }
  
  // 読込み
  MPI_Type_size(dataType[0], &type_size);
  if(type_size > 0){
     if (MPI_File_read(pstXmp_file->fh,
                       (*XMP_array_t->array_addr_p),
                       1,
                       dataType[0],
                       &status)
         != MPI_SUCCESS)
        {
           return -1;
        }
  } else {
     return 0;
  }
  
  // 使用しなくなったMPI_Datatypeを解放
  MPI_Type_free(&dataType[0]);

  // 読込んだバイト数
  if (MPI_Get_count(&status, MPI_BYTE, &readCount) != MPI_SUCCESS)
  {
    return -1;
  }
  return readCount;
}

/*****************************************************************************/
/*  FUNCTION NAME : xmp_fwrite_darray_pack                                   */
/*  DESCRIPTION   : この関数はapで指定される分散配列について、rpで指定される */
/*                  範囲からファイルへデータを書込む。                       */
/*                  派生型は生成せずにパックする。                           */
/*                  この関数は集団実行しなければならない。                   */
/*  ARGUMENT      : fp[IN] ファイル構造体                                    */
/*                  ap[IN] 分散配列情報                                      */
/*                  rp[IN]     アクセス範囲情報                              */
/*  RETURN VALUES : 正常終了の場合は書込んだバイト数を返す。                 */
/*                  異常終了の場合は負数を返す。                             */
/*                                                                           */
/*****************************************************************************/
int xmp_fwrite_darray_pack(fp, ap, rp)
     xmp_file_t *fp;
     xmp_array_t ap;
     xmp_range_t *rp;
{
   MPI_Status    status;
   _XMP_array_t *array_t;
   char         *array_addr;
   char         *buf=NULL;
   char         *cp;
   int          *lb=NULL;
   int          *ub=NULL;
   int          *step=NULL;
   int          *cnt=NULL;
   int           buf_size;
   int           ret=0;
   int           disp;
   int           size;
   int           array_size;
   int           i, j;

   array_t = (_XMP_array_t*)ap;
  
   /* 回転数を示す配列確保 */
   lb = (int*)malloc(sizeof(int)*rp->dims);
   ub = (int*)malloc(sizeof(int)*rp->dims);
   step = (int*)malloc(sizeof(int)*rp->dims);
   cnt = (int*)malloc(sizeof(int)*rp->dims);
   if(!lb || !ub || !step || !cnt){
      ret = -1;
      goto FunctionExit;
   }
  
   /* 回転数を求める */
   buf_size = 1;
   for(i=0; i<rp->dims; i++){
      /* error check */
      if(rp->step[i] > 0 && rp->lb[i] > rp->ub[i]){
         ret = -1;
         goto FunctionExit;
      }
      if(rp->step[i] < 0 && rp->lb[i] < rp->ub[i]){
         ret = -1;
         goto FunctionExit;
      }
      if (array_t->info[i].align_manner == _XMP_N_ALIGN_NOT_ALIGNED ||
          array_t->info[i].align_manner == _XMP_N_ALIGN_DUPLICATION) {
         lb[i] = rp->lb[i];
         ub[i] = rp->ub[i];
         step[i] = rp->step[i];
  
      } else if(array_t->info[i].align_manner == _XMP_N_ALIGN_BLOCK){
         if(rp->step[i] > 0){
            if(array_t->info[i].par_upper < rp->lb[i] ||
               array_t->info[i].par_lower > rp->ub[i]){
               lb[i] = 1;
               ub[i] = 0;
               step[i] = 1;
            } else {
               lb[i] = (array_t->info[i].par_lower > rp->lb[i])?
                  rp->lb[i]+((array_t->info[i].par_lower-1-rp->lb[i])/rp->step[i]+1)*rp->step[i]:
                  rp->lb[i];
               ub[i] = (array_t->info[i].par_upper < rp->ub[i]) ?
                  array_t->info[i].par_upper:
                  rp->ub[i];
               step[i] = rp->step[i];
            }
         } else {
            if(array_t->info[i].par_upper < rp->ub[i] ||
               array_t->info[i].par_lower > rp->lb[i]){
               lb[i] = 1;
               ub[i] = 0;
               step[i] = 1;
            } else {
               lb[i] = (array_t->info[i].par_upper < rp->lb[i])?
                  rp->lb[i]-((rp->lb[i]-array_t->info[i].par_upper-1)/rp->step[i]-1)*rp->step[i]:
                  rp->lb[i];
               ub[i] = (array_t->info[i].par_lower > rp->ub[i])?
                  array_t->info[i].par_lower:
                  rp->ub[i];
               step[i] = rp->step[i];
            }
         }
      } else {
         ret = -1;
         goto FunctionExit;
      }
      cnt[i] = (ub[i]-lb[i]+step[i])/step[i];
      cnt[i] = (cnt[i]>0)? cnt[i]: 0;
      buf_size *= cnt[i];

#ifdef DEBUG
      fprintf(stderr, "dim = %d: (%d: %d: %d) %d\n", i, lb[i], ub[i], step[i], buf_size);
#endif
   }
  
   /* バッファ確保 */
   if(buf_size == 0){
      buf = (char*)malloc(array_t->type_size);
      fprintf(stderr, "size = 0\n");
   } else {
      buf = (char*)malloc(buf_size*array_t->type_size);
   }
   if(!buf){
      ret = -1;
      goto FunctionExit;
   }

   /* データをパック */
   cp = buf;
   array_addr = (char*)(*array_t->array_addr_p);
   for(j=0; j<buf_size; j++){
      disp = 0;
      size = 1;
      array_size = 1;
      for(i=rp->dims-1; i>=0; i--){
         ub[i] = (j/size)%cnt[i];
         if (array_t->info[i].align_manner == _XMP_N_ALIGN_NOT_ALIGNED ||
             array_t->info[i].align_manner == _XMP_N_ALIGN_DUPLICATION) {
            disp += (lb[i]+ub[i]*step[i])*array_size;
            array_size *= array_t->info[i].ser_size;
         } else if(array_t->info[i].align_manner == _XMP_N_ALIGN_BLOCK){
            disp += (lb[i]+ub[i]*step[i]+array_t->info[i].local_lower-array_t->info[i].par_lower)*array_size;
            array_size *= array_t->info[i].alloc_size;
         }
         size *= cnt[i];
      }
      disp *= array_t->type_size;
      memcpy(cp, array_addr+disp, array_t->type_size);
      cp += array_t->type_size;
   }

  // 書込み
   if(buf_size > 0){
      if (MPI_File_write(fp->fh, buf, buf_size*array_t->type_size, MPI_BYTE, &status) != MPI_SUCCESS) {
         ret = -1;
         goto FunctionExit;
      }
      
      // 書き込んだバイト数
      if (MPI_Get_count(&status, MPI_BYTE, &ret) != MPI_SUCCESS) {
         ret = -1;
         goto FunctionExit;
      }
   } else {
      ret = 0;
   }
  
 FunctionExit:
   if(buf) free(buf);
   if(lb) free(lb);
   if(ub) free(ub);
   if(step) free(step);
   if(cnt) free(cnt);

   return ret;
}

/*****************************************************************************/
/*  FUNCTION NAME : xmp_fwrite_darray_all                                    */
/*  DESCRIPTION   : この関数はapで指定される分散配列について、rpで指定される */
/*                  範囲のデータをファイルに書込む。                         */
/*                  この関数は集団実行しなければならない。                   */
/*  ARGUMENT      : pstXmp_file[IN] ファイル構造体                           */
/*                  ap[IN/OUT] 分散配列情報                                  */
/*                  rp[IN]     アクセス範囲情報                              */
/*  RETURN VALUES : 正常終了の場合は読込んだバイト数を返す。                 */
/*                  異常終了の場合は負数を返す。                             */
/*                                                                           */
/*****************************************************************************/
size_t xmp_fwrite_darray_all(xmp_file_t *pstXmp_file,
                             xmp_array_t ap,
                             xmp_range_t *rp)
{
  _XMP_array_t *XMP_array_t;
  MPI_Status status;        // MPIステータス
  int writeCount;           // ライトバイト
  int mpiRet;               // MPI関数戻り値
  int lower;                // このノードでアクセスする下限
  int upper;                // このノードでアクセスする上限
  int continuous_size;      // 連続域サイズ
  int space_size;           // 隙間サイズ
  int total_size;           // 全体サイズ
  int type_size;
  MPI_Aint tmp1, tmp2;
  MPI_Datatype dataType[2];
  int i = 0;
#ifdef DEBUG
  int rank, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
#endif

  // 引数チェック
  if (pstXmp_file == NULL) { return -1; }
  if (ap == NULL)          { return -1; }
  if (rp == NULL)          { return -1; }

  XMP_array_t = (_XMP_array_t*)ap;

  // 次元数のチェック
  if (XMP_array_t->dim != rp->dims) { return -1; }

#ifdef DEBUG
printf("WRITE(%d/%d) dims=%d\n",rank, nproc, rp->dims);
#endif

  /* パックが必要な場合は別処理 */
  for (i = rp->dims - 1; i >= 0; i--){
     if(rp->step[i] < 0){
        int ret = xmp_fwrite_darray_pack(pstXmp_file, ap, rp);
        return ret;
     }
  }

  // 基本データ型の作成
  MPI_Type_contiguous(XMP_array_t->type_size, MPI_BYTE, &dataType[0]);

  // 次元数分ループ
  for (i = rp->dims - 1; i >= 0; i--)
  {
#ifdef DEBUG
printf("WRITE(%d/%d) (lb,ub,step)=(%d,%d,%d)\n",
       rank, nproc, rp->lb[i],  rp->ub[i], rp->step[i]);
printf("WRITE(%d/%d) (par_lower,par_upper, par_size)=(%d,%d,%d)\n",
       rank, nproc, XMP_array_t->info[i].par_lower,
       XMP_array_t->info[i].par_upper, XMP_array_t->info[i].par_size);
printf("WRITE(%d/%d) (local_lower,local_upper,alloc_size)=(%d,%d,%d)\n",
       rank, nproc, XMP_array_t->info[i].local_lower,
        XMP_array_t->info[i].local_upper,  XMP_array_t->info[i].alloc_size);
printf("WRITE(%d/%d) (shadow_size_lo,shadow_size_hi)=(%d,%d)\n",
       rank, nproc, XMP_array_t->info[i].shadow_size_lo, XMP_array_t->info[i].shadow_size_hi);
#endif

    // 分散の無い次元
    if (XMP_array_t->info[i].align_manner == _XMP_N_ALIGN_NOT_ALIGNED ||
        XMP_array_t->info[i].align_manner == _XMP_N_ALIGN_DUPLICATION)
    {
      // 分割後上限 < 下限
      if (XMP_array_t->info[i].par_upper < rp->lb[i]) { return -1; }
      // 分割後下限 > 上限
      if (XMP_array_t->info[i].par_lower > rp->ub[i]) { return -1; }

      // 増分が負
      if ( rp->step[i] < 0)
      {
      }
      // 増分が正
      else
      {
        // 連続域のサイズ
        continuous_size = (rp->ub[i] - rp->lb[i]) / rp->step[i] + 1;

        // データ型の範囲を取得
        mpiRet =MPI_Type_get_extent(dataType[0], &tmp1, &tmp2);
        if (mpiRet !=  MPI_SUCCESS) { return -1; }  
        type_size = (int)tmp2;

        // 基本データ型の生成
        mpiRet = MPI_Type_create_hvector(continuous_size,
                                         1,
                                         type_size * rp->step[i],
                                         dataType[0],
                                         &dataType[1]);

        // 使用しなくなったMPI_Datatypeを解放
        MPI_Type_free(&dataType[0]);

        // MPI_Type_contiguousがエラーの場合
        if (mpiRet != MPI_SUCCESS) { return -1; }

        // 全体サイズ
        total_size
          = (XMP_array_t->info[i].ser_upper
          -  XMP_array_t->info[i].ser_lower + 1)
          *  type_size;

        // 隙間サイズ
        space_size
          = (rp->lb[i] - XMP_array_t->info[i].par_lower)
          * type_size;

        // 新しいファイル型の作成
        mpiRet = MPI_Type_create_resized(dataType[1],
                                         (MPI_Aint)space_size,
                                         (MPI_Aint)total_size,
                                         &dataType[0]);

        // MPI_Type_create_resizedがエラーの場合
        if (mpiRet != MPI_SUCCESS) { return -1; }

        // 使用しなくなったMPI_Datatypeを解放
        MPI_Type_free(&dataType[1]);

#ifdef DEBUG
printf("WRITE(%d/%d) NOT_ALIGNED\n",rank, nproc);
printf("WRITE(%d/%d) type_size=%d\n",rank, nproc, type_size);
printf("WRITE(%d/%d) continuous_size=%d\n",rank, nproc, continuous_size);
printf("WRITE(%d/%d) space_size=%d\n",rank, nproc, space_size);
printf("WRITE(%d/%d) total_size=%d\n",rank, nproc, total_size);
#endif
      }
    }
     // block分散
    else if (XMP_array_t->info[i].align_manner == _XMP_N_ALIGN_BLOCK)
    {
      // 増分が負
      if ( rp->step[i] < 0)
      {
      }
      // 増分が正
      else
      {
        // 分割後上限 < 下限
        if (XMP_array_t->info[i].par_upper < rp->lb[i])
        {
          continuous_size = 0;
        }
        // 分割後下限 > 上限
        else if (XMP_array_t->info[i].par_lower > rp->ub[i])
        {
          continuous_size = 0;
        }
        // その他
        else
        {
          // ノードの下限
          lower
            = (XMP_array_t->info[i].par_lower > rp->lb[i]) ?
              rp->lb[i] + ((XMP_array_t->info[i].par_lower - 1 - rp->lb[i]) / rp->step[i] + 1) * rp->step[i]
            : rp->lb[i];

          // ノードの上限
          upper
            = (XMP_array_t->info[i].par_upper < rp->ub[i]) ?
               XMP_array_t->info[i].par_upper : rp->ub[i];

          // 連続要素数
          continuous_size = (upper - lower + rp->step[i]) / rp->step[i];
        }

        // データ型の範囲を取得
        mpiRet =MPI_Type_get_extent(dataType[0], &tmp1, &tmp2);
        if (mpiRet !=  MPI_SUCCESS) { return -1; }  
        type_size = (int)tmp2;
        if(lower > upper){
           type_size = 0;
        }

        // 基本データ型の生成
        mpiRet = MPI_Type_create_hvector(continuous_size,
                                         1,
                                         type_size * rp->step[i],
                                         dataType[0],
                                         &dataType[1]);

        // 使用しなくなったMPI_Datatypeを解放
        MPI_Type_free(&dataType[0]);

        // MPI_Type_create_hvectorがエラーの場合
        if (mpiRet != MPI_SUCCESS) { return -1; }

        // 隙間サイズ
        space_size
          = (XMP_array_t->info[i].local_lower
          + (lower - XMP_array_t->info[i].par_lower))
          * type_size;

        // 全体サイズ
        total_size = (XMP_array_t->info[i].alloc_size)* type_size;

        // 新しいファイル型の作成
        mpiRet = MPI_Type_create_resized(dataType[1],
                                         (MPI_Aint)space_size,
                                         (MPI_Aint)total_size,
                                         &dataType[0]);

        // MPI_Type_create_resizedがエラーの場合
        if (mpiRet != MPI_SUCCESS) { return -1; }

        // 使用しなくなったMPI_Datatypeを解放
        MPI_Type_free(&dataType[1]);

#ifdef DEBUG
printf("WRITE(%d/%d) ALIGN_BLOCK\n",rank, nproc);
printf("WRITE(%d/%d) type_size=%d\n",rank, nproc, type_size);
printf("WRITE(%d/%d) continuous_size=%d\n",rank, nproc, continuous_size);
printf("WRITE(%d/%d) space_size=%d\n",rank, nproc, space_size);
printf("WRITE(%d/%d) total_size=%d\n",rank, nproc, total_size);
printf("WRITE(%d/%d) (lower,upper)=(%d,%d)\n",rank, nproc, lower, upper);
#endif
      }
    }
    // cyclic分散
    else if (XMP_array_t->info[i].align_manner == _XMP_N_ALIGN_CYCLIC)
    {
      return -1;
    }
    // その他
    else
    {
      return -1;
    }
  }

  // コミット
  mpiRet = MPI_Type_commit(&dataType[0]);

  // コミットがエラーの場合
  if (mpiRet != MPI_SUCCESS) { return 1; }
 
  // 書込み
  MPI_Type_size(dataType[0], &type_size);
  if(type_size > 0){
     if (MPI_File_write(pstXmp_file->fh,
                        (*XMP_array_t->array_addr_p),
                        1,
                        dataType[0],
                        &status)
         != MPI_SUCCESS)
        {
           return -1;
        }
  } else {
     return 0;
  }
 
  // 使用しなくなったMPI_Datatypeを解放
  MPI_Type_free(&dataType[0]);

  // 読込んだバイト数
  if (MPI_Get_count(&status, MPI_BYTE, &writeCount) != MPI_SUCCESS)
  {
    return -1;
  }
  return writeCount;
}

/*****************************************************************************/
/*  FUNCTION NAME : xmp_fread_shared                                         */
/*  DESCRIPTION   : この関数は実行したノードの値をファイルの共有ファイル     */
/*                  ポインタの位置から読込む。                               */
/*                  この関数はローカル実行可能である。                       */
/*  ARGUMENT      : pstXmp_file[IN] ファイル構造体                           */
/*                  buffer[OUT] 非分散変数の先頭アドレス                     */
/*                  size[IN]  読込むデータの1要素当りのサイズ (バイト)       */
/*                  count[IN] 読込むデータの数                               */
/*  RETURN VALUES : 正常終了の場合は読込んだバイト数を返す。                 */
/*                  異常終了の場合は負数を返す。                             */
/*****************************************************************************/
size_t xmp_fread_shared(xmp_file_t *pstXmp_file, void *buffer, size_t size, size_t count)
{
  MPI_Status status;
  int readCount;

  // 引数チェック
  if (pstXmp_file == NULL) { return -1; }
  if (buffer      == NULL) { return -1; }
  if (size  < 1) { return -1; }
  if (count < 1) { return -1; }

  // 読込み
  if (MPI_File_read_shared(pstXmp_file->fh, buffer, size * count, MPI_BYTE, &status) != MPI_SUCCESS)
  {
    return -1;
  }
  
  // 読込んだバイト数
  if (MPI_Get_count(&status, MPI_BYTE, &readCount) != MPI_SUCCESS)
  {
    return -1;
  }

  return readCount;
}

/*****************************************************************************/
/*  FUNCTION NAME : xmp_fwrite_shared                                        */
/*  DESCRIPTION   : この関数は実行したノードの値をファイルの共有ファイル     */
/*                  ポインタの位置に書込む。                                 */
/*                  この関数はローカル実行可能である。                       */
/*  ARGUMENT      : pstXmp_file[IN] ファイル構造体                           */
/*                  buffer[IN] 非分散変数の先頭アドレス                      */
/*                  size[IN]   書込むデータの1要素当りのサイズ (バイト)      */
/*                  count[IN]  書込むデータの数                              */
/*  RETURN VALUES : 正常終了の場合は書込んだバイト数を返す。                 */
/*                  異常終了の場合は負数を返す。                             */
/*                                                                           */
/*****************************************************************************/
size_t xmp_fwrite_shared(xmp_file_t *pstXmp_file, void *buffer, size_t size, size_t count)
{
  MPI_Status status;
  int writeCount;

  // 引数チェック
  if (pstXmp_file == NULL) { return -1; }
  if (buffer      == NULL) { return -1; }
  if (size  < 1) { return -1; }
  if (count < 1) { return -1; }

  // ファイルオープンが"r+"の場合は終端にポインタを移動
  if (pstXmp_file->is_append)
  {
    if (MPI_File_seek_shared(pstXmp_file->fh,
                             (MPI_Offset)0,
                             MPI_SEEK_END) != MPI_SUCCESS)
    {
      return -1;
    }

    pstXmp_file->is_append = 0x00;
  }

  // 書込み
  if (MPI_File_write_shared(pstXmp_file->fh,
                            buffer,
                            size * count,
                            MPI_BYTE,
                            &status) != MPI_SUCCESS)
  {
    return -1;
  }

  // 書込んだバイト数
  if (MPI_Get_count(&status, MPI_BYTE, &writeCount) != MPI_SUCCESS)
  {
    return -1;
  }

  return writeCount;
}

/*****************************************************************************/
/*  FUNCTION NAME : xmp_fread                                                */
/*  DESCRIPTION   : この関数は実行したノードのbufferへファイルビューに従い   */
/*                  データを読込む。                                         */
/*                  この関数はローカル実行可能である。                       */
/*  ARGUMENT      : pstXmp_file[IN] ファイル構造体                           */
/*                  buffer[OUT] 非分散変数の先頭アドレス                     */
/*                  size[IN]    読込むデータの1要素当りのサイズ (バイト)     */
/*                  count[IN]   読込むデータの数                             */
/*  RETURN VALUES : 正常終了の場合は読込んだバイト数を返す。                 */
/*                  異常終了の場合は負数を返す。                             */
/*                                                                           */
/*****************************************************************************/
size_t xmp_fread(xmp_file_t *pstXmp_file, void *buffer, size_t size, size_t count)
{
  MPI_Status status;
  int readCount;

  // 引数チェック
  if (pstXmp_file == NULL) { return -1; }
  if (buffer      == NULL) { return -1; }
  if (size  < 1) { return -1; }
  if (count < 1) { return -1; }

  // 読込み
  if (MPI_File_read(pstXmp_file->fh, buffer, size * count, MPI_BYTE, &status) != MPI_SUCCESS)
  {
    return -1;
  }
  
  // 読込んだバイト数
  if (MPI_Get_count(&status, MPI_BYTE, &readCount) != MPI_SUCCESS)
  {
    return -1;
  }

  return readCount;
}

/*****************************************************************************/
/*  FUNCTION NAME : xmp_fwrite                                               */
/*  DESCRIPTION   : この関数は実行したノードのbufferからファイルビューに     */
/*                  従いデータを書込む。                                     */
/*                  この関数はローカル実行可能である。                       */
/*  ARGUMENT      : pstXmp_file[IN] ファイル構造体                           */
/*                  buffer[IN] 非分散変数の先頭アドレス                      */
/*                  size[IN]   書込むデータの1要素当りのサイズ (バイト)      */
/*                  count[IN]  書込むデータの数                              */
/*  RETURN VALUES : 正常終了の場合は書込んだバイト数を返す。                 */
/*                  異常終了の場合は負数を返す。                             */
/*                                                                           */
/*****************************************************************************/
size_t xmp_fwrite(xmp_file_t *pstXmp_file, void *buffer, size_t size, size_t count)
{
  MPI_Status status;
  int writeCount;

  // 引数チェック
  if (pstXmp_file == NULL) { return -1; }
  if (buffer      == NULL) { return -1; }
  if (size  < 1) { return -1; }
  if (count < 1) { return -1; }

  // ファイルオープンが"r+"の場合は終端にポインタを移動
  if (pstXmp_file->is_append)
  {
    if (MPI_File_seek(pstXmp_file->fh,
                      (MPI_Offset)0,
                      MPI_SEEK_END) != MPI_SUCCESS)
    {
      return -1;
    }

    pstXmp_file->is_append = 0x00;
  }

  // 書込み
  if (MPI_File_write(pstXmp_file->fh,
                     buffer,
                     size * count,
                     MPI_BYTE,
                     &status) != MPI_SUCCESS)
  {
    return -1;
  }

  // 書込んだバイト数
  if (MPI_Get_count(&status, MPI_BYTE, &writeCount) != MPI_SUCCESS)
  {
    return -1;
  }

  return writeCount;
}


/*****************************************************************************/
/*  FUNCTION NAME : xmp_file_set_view                                        */
/*  DESCRIPTION   : この関数はapで指定される分散配列について、rpで指定される */
/*                  範囲のファイルビューをfhへ設定する。                     */
/*                  この関数は集団実行しなければならない。                   */
/*  ARGUMENT      : pstXmp_file[IN] ファイル構造体                           */
/*                  disp[IN] ファイル先頭からの変位 (バイト)                 */
/*                  ap[IN]   分散配列情報                                    */
/*                  rp[IN]   アクセス範囲情報                                */
/*  RETURN VALUES : 正常終了の場合は0を返す。                                */
/*                  異常終了の場合は0以外の値を返す。                        */
/*                                                                           */
/*****************************************************************************/
int xmp_file_set_view_all(xmp_file_t  *pstXmp_file,
                          long long    disp,
                          xmp_array_t  ap,
                          xmp_range_t *rp)
{
  _XMP_array_t *XMP_array_t;
  int i = 0;
  int mpiRet;               // MPI関数戻り値
  int lower;                // このノードでアクセスする下限
  int upper;                // このノードでアクセスする上限
  int continuous_size;      // 連続域サイズ
  MPI_Datatype dataType[2];
  int type_size;
  MPI_Aint tmp1, tmp2;
#ifdef DEBUG
  int rank, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
#endif

  // 引数チェック
  if (pstXmp_file == NULL) { return 1; }
  if (ap == NULL)          { return 1; }
  if (rp == NULL)          { return 1; }
  if (disp  < 0)           { return 1; }

  XMP_array_t = (_XMP_array_t*)ap; 

  // 次元数のチェック
  if (XMP_array_t->dim != rp->dims) { return 1; }

#ifdef DEBUG
printf("VIEW(%d/%d) dims=%d\n", rank, nproc, rp->dims);
#endif

  // 基本データ型の作成
  MPI_Type_contiguous(XMP_array_t->type_size, MPI_BYTE, &dataType[0]);

  // 次元数分ループ
  for (i = rp->dims - 1; i >= 0; i--)
  {
    // データ型の範囲を取得
    mpiRet =MPI_Type_get_extent(dataType[0], &tmp1, &tmp2);
    if (mpiRet !=  MPI_SUCCESS) { return -1; }  
    type_size = (int)tmp2;

#ifdef DEBUG
printf("VIEW(%d/%d) (lb,ub,step)=(%d,%d,%d)\n",
        rank, nproc, rp->lb[i],  rp->ub[i], rp->step[i]);
printf("VIEW(%d/%d) (par_lower,par_upper)=(%d,%d)\n",
        rank, nproc, XMP_array_t->info[i].par_lower, XMP_array_t->info[i].par_upper);
#endif
    // 分散の無い次元
    if (XMP_array_t->info[i].align_manner == _XMP_N_ALIGN_NOT_ALIGNED ||
        XMP_array_t->info[i].align_manner == _XMP_N_ALIGN_DUPLICATION)
    {
      // 連続域のサイズ
      continuous_size = (rp->ub[i] - rp->lb[i]) / rp->step[i] + 1;

      // 基本データ型の生成
      mpiRet = MPI_Type_contiguous(continuous_size, dataType[0], &dataType[1]);

      // 使用しなくなったMPI_Datatypeを解放
      MPI_Type_free(&dataType[0]);
      dataType[0] = dataType[1];

      // MPI_Type_contiguousがエラーの場合
      if (mpiRet != MPI_SUCCESS) { return 1; }

#ifdef DEBUG
printf("VIEW(%d/%d) NOT_ALIGNED\n", rank, nproc);
printf("VIEW(%d/%d) continuous_size=%d\n", rank, nproc, continuous_size);
#endif
    }
    // block分散
    else if (XMP_array_t->info[i].align_manner == _XMP_N_ALIGN_BLOCK)
    {
      int space_size;
      int total_size;

      // 増分が正の場合
      if (rp->step[i] >= 0)
      {
        // 下限 > 上限
        if (rp->lb[i] > rp->ub[i])
        {
          return 1;
        }
        // 分割後上限 < 下限
        else if (XMP_array_t->info[i].par_upper < rp->lb[i])
        {
          continuous_size = space_size = 0;
        }
        // 分割後下限 > 上限
        else if (XMP_array_t->info[i].par_lower > rp->ub[i])
        {
          continuous_size = space_size = 0;
        }
        // その他
        else
        {
          // ノードの下限
          lower
            = (XMP_array_t->info[i].par_lower > rp->lb[i]) ?
              rp->lb[i] + ((XMP_array_t->info[i].par_lower - 1 - rp->lb[i]) / rp->step[i] + 1) * rp->step[i]
            : rp->lb[i];

          // ノードの上限
          upper
            = (XMP_array_t->info[i].par_upper < rp->ub[i]) ?
               XMP_array_t->info[i].par_upper : rp->ub[i];

          // 連続要素数
          continuous_size = (upper - lower) / rp->step[i] + 1;

          // 隙間サイズ
          space_size
            = ((lower - rp->lb[i]) / rp->step[i]) * type_size;
        }

        // 全体サイズ
        total_size
          = ((rp->ub[i] - rp->lb[i]) / rp->step[i] + 1) * type_size;

        // 基本データ型の生成
        mpiRet = MPI_Type_contiguous(continuous_size, dataType[0], &dataType[1]);

        // 使用しなくなったMPI_Datatypeを解放
        MPI_Type_free(&dataType[0]);

        // MPI_Type_contiguousがエラーの場合
        if (mpiRet != MPI_SUCCESS) { return 1; }

        // 新しいファイル型の作成
        mpiRet = MPI_Type_create_resized(dataType[1],
                                         space_size,
                                         total_size,
                                         &dataType[0]);

        // MPI_Type_create_resizedがエラーの場合
        if (mpiRet != MPI_SUCCESS) { return 1; }


        // 使用しなくなったMPI_Datatypeを解放
        MPI_Type_free(&dataType[1]);

#ifdef DEBUG
printf("VIEW(%d/%d) ALIGN_BLOCK\n", rank, nproc );
printf("VIEW(%d/%d) type_size=%d\n", rank, nproc , type_size);
printf("VIEW(%d/%d) continuous_size=%d\n", rank, nproc , continuous_size);
printf("VIEW(%d/%d) space_size=%d\n", rank, nproc , space_size);
printf("VIEW(%d/%d) total_size=%d\n", rank, nproc , total_size);
printf("VIEW(%d/%d) (lower,upper)=(%d,%d)\n", rank, nproc , lower, upper);
printf("\n");
#endif
      }
      // 増分が負の場合
      else if (rp->step[i] < 0)
      {
        // 下限 < 上限
        if (rp->lb[i] < rp->ub[i])
        {
          return 1;
        }
        // 分割後下限 < 上限
        else if (XMP_array_t->info[i].par_lower < rp->ub[i])
        {
          continuous_size = space_size = 0;
        }
        // 分割後上限 > 下限
        else if (XMP_array_t->info[i].par_upper > rp->lb[i])
        {
          continuous_size = space_size = 0;
        }
        // その他
        else
        {
          // ノードの下限
          lower
            = (XMP_array_t->info[i].par_upper <  rp->lb[i]) ?
              rp->lb[i] - (( rp->lb[i] - XMP_array_t->info[i].par_upper - 1) / rp->step[i] - 1) * rp->step[i]
            : rp->lb[i];

          // ノードの上限
          upper
            = (XMP_array_t->info[i].par_lower > rp->ub[i]) ?
               XMP_array_t->info[i].par_lower : rp->ub[i];

          // 連続要素数
          continuous_size = (upper - lower) / rp->step[i] + 1;

          // 隙間サイズ
          space_size
            = ((lower - rp->lb[i]) / rp->step[i]) * type_size;
        }

        // 基本データ型の生成
        mpiRet = MPI_Type_contiguous(continuous_size, dataType[0], &dataType[1]);

        // 全体サイズ
        total_size
          = ((rp->ub[i] - rp->lb[i]) / rp->step[i] + 1) * type_size;

        // 使用しなくなったMPI_Datatypeを解放
        MPI_Type_free(&dataType[0]);

        // MPI_Type_contiguousがエラーの場合
        if (mpiRet != MPI_SUCCESS) { return 1; }

        // 新しいファイル型の作成
        mpiRet = MPI_Type_create_resized(dataType[1],
                                         space_size,
                                         total_size,
                                         &dataType[0]);

        // MPI_Type_create_resizedがエラーの場合
        if (mpiRet != MPI_SUCCESS) { return 1; }

        // 使用しなくなったMPI_Datatypeを解放
        MPI_Type_free(&dataType[1]);

#ifdef DEBUG
printf("VIEW(%d/%d) ALIGN_BLOCK\n", rank, nproc);
printf("VIEW(%d/%d) continuous_size=%d\n", rank, nproc, continuous_size);
printf("VIEW(%d/%d) space_size=%d\n", rank, nproc, space_size);
printf("VIEW(%d/%d) total_size=%d\n", rank, nproc, total_size);
printf("VIEW(%d/%d) (lower,upper)=(%d,%d)\n", rank, nproc, lower, upper);
#endif
      }
    }
    // cyclic分散
    else if (XMP_array_t->info[i].align_manner == _XMP_N_ALIGN_CYCLIC)
    {
      return 1;
    }
    // その他
    else
    {
      return 1;
    }
  }

  // コミット
  mpiRet = MPI_Type_commit(&dataType[0]);

  // コミットがエラーの場合
  if (mpiRet != MPI_SUCCESS) { return 1; }
  
  // ビューのセット
  mpiRet = MPI_File_set_view(pstXmp_file->fh,
                             (MPI_Offset)disp,
                             MPI_BYTE,
                             dataType[0],
                             "native",
                             MPI_INFO_NULL);


  // 使用しなくなったMPI_Datatypeを解放
  //MPI_Type_free(&dataType[0]);

  // ビューのセットがエラーの場合
  if (mpiRet != MPI_SUCCESS) { return 1; }

  return 0;
}

/*****************************************************************************/
/*  FUNCTION NAME : xmp_file_clear_view                                      */
/*  DESCRIPTION   : この関数はファイルビューを初期化する。初期化されると、   */
/*                  各ファイルポインタはdispに設定され、要素データ型と       */
/*                  ファイル型はMPI_BYTEに設定される。                       */
/*                  この関数は集団実行しなければならない。                   */
/*  ARGUMENT      : pstXmp_file[IN] ファイル構造体                           */
/*                  disp[IN] ファイル先頭からの変位 (バイト)                 */
/*  RETURN VALUES : 正常終了の場合は0を返す。                                */
/*                  異常終了の場合は0以外の値を返す。                        */
/*                                                                           */
/*****************************************************************************/
int xmp_file_clear_view_all(xmp_file_t  *pstXmp_file, long long disp)
{
  // 引数チェック
  if (pstXmp_file == NULL) { return 1; }
  if (disp  < 0)           { return 1; }

  // ビューの初期化
  if (MPI_File_set_view(pstXmp_file->fh,
                        disp,
                        MPI_BYTE,
                        MPI_BYTE,
                        "native",
                        MPI_INFO_NULL) != MPI_SUCCESS)
  {
    return 1;
  }

  return 0;
}

/*****************************************************************************/
/*  FUNCTION NAME : MPI_Type_create_resized                                  */
/*                                                                           */
/*****************************************************************************/
int MPI_Type_create_resized(MPI_Datatype oldtype,
                            MPI_Aint     lb,
                            MPI_Aint     extent,
                            MPI_Datatype *newtype)
{
        int          mpiRet;
        int          b[3];
        MPI_Aint     d[3];
        MPI_Datatype t[3];

        b[0] = b[1] = b[2] = 1;
        d[0] = 0;
        d[1] = lb;
        d[2] = extent;
        t[0] = MPI_LB;
        t[1] = oldtype;
        t[2] = MPI_UB;

        mpiRet = MPI_Type_create_struct(3, b, d, t, newtype);

        return mpiRet;
}
