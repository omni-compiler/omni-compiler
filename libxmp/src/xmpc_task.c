#include "xmp_internal.h"

void xmpc_create_task_nodes(_XMP_nodes_t **n, _XMP_object_ref_t *r_desc)
{
  _XMP_create_task_nodes(n, r_desc);
}


_Bool xmpc_test_task_on_nodes(_XMP_nodes_t *n, _XMP_object_ref_t *r_desc)
{
  int is_active = _XMP_test_task_on_nodes(n);
#ifdef _XMPT
  xmpt_tool_data_t *data = &((*n)->xmpt_data); *data=NULL;
  xmp_desc_t on = r_desc->ref_kind == XMP_OBJ_REF_NODES ?
    (xmp_desc_t)r_desc->n_desc : (xmp_desc_t)r_desc->t_desc;
  struct _xmpt_subscript_t on_subsc;
  _XMPT_set_subsc(&on_subsc, r_desc);
  if (xmpt_enabled && xmpt_callback[xmpt_event_task_begin]){
    (*(xmpt_event_task_begin_t)xmpt_callback[xmpt_event_task_begin])(
	on,
	&on_subsc,
	is_active,
	data);
  }
#endif
  return is_active;
}


_Bool xmpc_test_task_nocomm(_XMP_object_ref_t *r_desc)
{
  int is_active = _XMP_test_task_nocomm(r_desc);
#ifdef _XMPT
  xmpt_tool_data_t *data = &((*n)->xmpt_data); *data=NULL;
  xmp_desc_t on = r_desc->ref_kind == XMP_OBJ_REF_NODES ?
    (xmp_desc_t)r_desc->n_desc : (xmp_desc_t)r_desc->t_desc;
  struct _xmpt_subscript_t on_subsc;
  _XMPT_set_subsc(&on_subsc, r_desc);
  if (xmpt_enabled && xmpt_callback[xmpt_event_task_begin]){
    (*(xmpt_event_task_begin_t)xmpt_callback[xmpt_event_task_begin])(
	on,
	&on_subsc,
	is_active,
	data);
  }
#endif
  return is_active;
}


void xmpc_end_task(void)
{
#ifdef _XMPT
  xmpt_tool_data_t *data = &(n->xmpt_data);
  if (xmpt_enabled && xmpt_callback[xmpt_event_task_end])
    (*(xmpt_event_end_t)xmpt_callback[xmpt_event_task_end])(data);
#endif
  _XMP_end_task();
}

void xmpc_finalize_task_nodes(_XMP_nodes_t *n)
{
  _XMP_finalize_nodes(n);
}
