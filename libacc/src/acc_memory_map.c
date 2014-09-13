#include <stdio.h>
#include <stdlib.h>
#include "../include/acc_internal.h"

typedef struct Node{
  void *addr;
  size_t size;
  void *data;
  struct Node *next;
}Node;


struct _ACC_memory_map_type{
  Node* head;
};

_ACC_memory_map_t* _ACC_memory_map_create()
{
  _ACC_memory_map_t* map = (_ACC_memory_map_t*)_ACC_alloc(sizeof(_ACC_memory_map_t));
  map->head = NULL;
  return map;
}

void _ACC_memory_map_destroy(_ACC_memory_map_t* map)
{
  while(map->head != NULL){
    Node *nextHeadNode = map->head->next;
    _ACC_free(map->head);
    map->head = nextHeadNode;
  }
  _ACC_free(map);
}

void _ACC_memory_map_add(_ACC_memory_map_t* map, void *addr, size_t size, void *data)
{
  Node *newNode = (Node*)_ACC_alloc(sizeof(Node));
  newNode->addr = addr;
  newNode->size = size;
  newNode->data = data;
  

  Node *oldHead = map->head;
  map->head = newNode;
  newNode->next = oldHead;
}

void* _ACC_memory_map_remove(_ACC_memory_map_t* map, void *addr, size_t size)
{
  Node *node, *prevNode = NULL;
  for(node = map->head; node != NULL; node = node->next){
    if(node->addr == addr && node->size == size){
      if(prevNode == NULL){
        map->head = node->next;
      }else{
        prevNode->next = node->next;
      }
      void* data = node->data;
      _ACC_free(node);
      return data;
    }
    prevNode = node;
  }
  return NULL;
}

void* _ACC_memory_map_find(_ACC_memory_map_t* map, void *addr, size_t size)
{
  Node *node;
  for(node = map->head; node != NULL; node = node->next){
    //if(node->addr == addr && node->size == size){
    if(node->addr <= addr && (char*)addr + size <= (char*)node->addr + node->size){
      return node->data;
    }
  }
  return NULL;
}

