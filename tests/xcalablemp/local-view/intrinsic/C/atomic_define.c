#include <xmp.h>
#include <stdio.h>
#include <stdlib.h>
int atom0:[*];
int atom1[3]:[*];
int atom2[3][3]:[*];
int value0:[*];
int value1[3]:[*];
int value2[3][3]:[*];
#define FALSE 0
#define TRUE  1
int flag;

void check()
{
#pragma xmp reduction(min:flag)
  if(flag == FALSE)
    exit(1);
  else
    if(xmp_node_num() == 1)
      printf("PASS\n");
}

void test1(int value)
{
  atom0 = -1;
  flag = TRUE;
  
  xmp_sync_all(NULL);
  if(xmp_node_num() == 2){
    xmp_atomic_define(atom0:[1], value);
  }
  xmp_sync_all(NULL);

  if(xmp_node_num() == 1)
    if(atom0 != 2)
      flag = FALSE;

  atom0 = -1;
  xmp_atomic_define(atom0, value);
  if(value != atom0)
    flag = FALSE;
}

void test1_c(int value)
{
  atom0 = -1;
  flag = TRUE;
  value0 = value;

  xmp_sync_all(NULL);
  if(xmp_node_num() == 2){
    xmp_atomic_define(atom0:[1], value0);
  }
  xmp_sync_all(NULL);

  if(xmp_node_num() == 1)
    if(atom0 != 2)
      flag = FALSE;

  atom0 = -1;
  xmp_atomic_define(atom0, value0);
  if(value0 != atom0)
    flag = FALSE;
}

void test2(int value)
{
  atom1[2] = -1;
  flag = TRUE;

  xmp_sync_all(NULL);
  if(xmp_node_num() == 2){
    xmp_atomic_define(atom1[2]:[1], value);
  }
  xmp_sync_all(NULL);

  if(xmp_node_num() == 1)
    if(atom1[2] != 2)
      flag = FALSE;

  atom1[1] = -1;
  xmp_atomic_define(atom1[1], value);
  if(value != atom1[1])
    flag = FALSE;
}

void test2_c(int value)
{
  atom1[2] = -1;
  flag = TRUE;
  value1[1] = value;

  xmp_sync_all(NULL);
  if(xmp_node_num() == 2){
    xmp_atomic_define(atom1[2]:[1], value1[1]);
  }
  xmp_sync_all(NULL);

  if(xmp_node_num() == 1)
    if(atom1[2] != 2)
      flag = FALSE;

  atom1[1] = -1;
  xmp_atomic_define(atom1[1], value1[1]);
  if(value1[1] != atom1[1])
    flag = FALSE;
}

void test3(int value)
{
  atom2[2][1] = -1;
  flag = TRUE;

  xmp_sync_all(NULL);
  if(xmp_node_num() == 2){
    xmp_atomic_define(atom2[2][1]:[1], value);
  }
  xmp_sync_all(NULL);

  if(xmp_node_num() == 1)
    if(atom2[2][1] != 2)
      flag = FALSE;

  atom2[2][2] = -1;
  xmp_atomic_define(atom2[2][2], value);
  if(value != atom2[2][2])
    flag = FALSE;
}

void test3_c(int value)
{
  atom2[2][1] = -1;
  flag = TRUE;
  value2[2][2] = value;

  xmp_sync_all(NULL);
  if(xmp_node_num() == 2){
    xmp_atomic_define(atom2[2][1]:[1], value2[2][2]);
  }
  xmp_sync_all(NULL);

  if(xmp_node_num() == 1)
    if(atom2[2][1] != 2)
      flag = FALSE;

  atom2[2][2] = -1;
  xmp_atomic_define(atom2[2][2], value2[2][2]);
  if(value2[2][2] != atom2[2][2])
    flag = FALSE;
}

int main()
{
  test1(xmp_node_num());   check();
  test1_c(xmp_node_num()); check();
  test2(xmp_node_num());   check();
  test2_c(xmp_node_num()); check();
  test3(xmp_node_num());   check();
  test3_c(xmp_node_num()); check();

  return 0;
}
