#include <xmp.h>
#include <stdio.h>
#include <stdlib.h>
int atom0:[*];
int atom1[3]:[*];
int atom2[3][3]:[*];
#define FALSE 0
#define TRUE  1
int flag;

void check()
{
#pragma xmp reduction(min:flag)
  if(flag == FALSE)
    exit(1);
}

void test1(int value)
{
  atom0 = -1;
  flag = TRUE;
  
  xmp_sync_all(NULL);
  if(xmp_node_num() == 2){
    xmp_atomic_ref(&value, atom0:[1]);
    if(value != -1)
      flag = FALSE;
  }
  xmp_sync_all(NULL);
  
  atom0 = -1;
  xmp_atomic_ref(&value, atom0);
  if(value != atom0)
    flag = FALSE;
  
  xmp_sync_all(NULL);
}

void test2(int value)
{
  atom1[2] = -1;
  flag = TRUE;

  xmp_sync_all(NULL);
  if(xmp_node_num() == 2){
    xmp_atomic_ref(&value, atom1[2]:[1]);
    if(atom1[2] != -1)
      flag = FALSE;
  }
  xmp_sync_all(NULL);
  
  atom1[1] = -1;
  xmp_atomic_ref(&value, atom1[1]);
  if(value != atom1[1])
    flag = FALSE;
  
  xmp_sync_all(NULL);
}

void test3(int value)
{
  atom2[2][1] = -1;
  flag = TRUE;

  xmp_sync_all(NULL);
  if(xmp_node_num() == 2){
    xmp_atomic_ref(&value, atom2[2][1]:[1]);
    if(atom2[2][1] != -1)
      flag = FALSE;
  }
  xmp_sync_all(NULL);
  
  atom2[2][2] = -1;
  xmp_atomic_ref(&value, atom2[2][2]);
  if(value != atom2[2][2])
    flag = FALSE;
  
  xmp_sync_all(NULL);
}

int main()
{
  test1(xmp_node_num()); check();
  test2(xmp_node_num()); check();
  test3(xmp_node_num()); check();
  
  return 0;
}
