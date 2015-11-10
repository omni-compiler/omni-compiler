#include <stdbool.h>
#include <stdio.h>
#include "acc_internal.h"

typedef struct _ACC_memory_list_type{
  char *addr;
  size_t size;
  _ACC_memory_t *memory;
  struct _ACC_memory_list_type *next;
} _ACC_memory_list_t;

static _ACC_memory_list_t *list_head = NULL;

void _ACC_gpu_init_data_table()
{
  _ACC_DEBUG("initialize\n")
  if(list_head != NULL){
    _ACC_gpu_finalize_data_table(); 
  }
}

void _ACC_gpu_finalize_data_table()
{
  _ACC_DEBUG("finalize\n")
  while(list_head != NULL){
    _ACC_memory_list_t *next = list_head->next;
    _ACC_memory_free(list_head->memory);
    _ACC_free(list_head);
    list_head = next;
  }
}

void _ACC_memory_table_add(void *host_addr, size_t size, _ACC_memory_t* memory)
{
  _ACC_DEBUG("add\n")

  _ACC_memory_list_t *new_element = (_ACC_memory_list_t*)_ACC_alloc(sizeof(_ACC_memory_list_t));
  new_element->addr = host_addr;
  new_element->size = size;
  new_element->memory = memory;
  new_element->next = list_head;
  list_head = new_element;
}

_ACC_memory_t* _ACC_memory_table_remove(void *addr, size_t size)
{
  _ACC_DEBUG("remove\n")
  _ACC_memory_list_t *prev_element = NULL;
  for(_ACC_memory_list_t *element = list_head; element != NULL; element = element->next){
    if(element->addr == addr && element->size == size){
      if(prev_element == NULL){ //current is head
	list_head = element->next;
      }else{
	prev_element->next = element->next;
      }
      _ACC_memory_t *memory = element->memory;
      _ACC_free(element);

      return memory;
    }
    prev_element = element;
  }
  return NULL;
}

_ACC_memory_t* _ACC_memory_table_find(void *addr, size_t size)
{
  _ACC_DEBUG("find\n")
  char *begin = (void*)((char*)addr);
  char *end = (void*)((char*)begin + size);

  for(_ACC_memory_list_t *element = list_head; element != NULL; element = element->next){
    char *memory_begin = element->addr;
    char *memory_end = memory_begin + element->size;

    if((memory_begin <= begin) && (end <= memory_end)){
      return element->memory;
    }
  }
  return NULL;
}
