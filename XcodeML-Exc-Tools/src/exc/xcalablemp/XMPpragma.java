package exc.xcalablemp;

import exc.object.Xobject;

/**
 * XcalableMP pragma codes
 */
public enum XMPpragma {
  // directive
  NODES,
  TEMPLATE,
  DISTRIBUTE,
  ALIGN,
  SHADOW,
  STATIC_DESC,
  TASK,
  TASKS,
  LOOP,
  REFLECT,
  GMOVE,
  BARRIER,
  REDUCTION,
  BCAST,
  COARRAY,
  ARRAY,
  //  SYNC_MEMORY,
  //  SYNC_ALL,
  POST,
  WAIT,
  LOCK,
  UNLOCK,
  LOCAL_ALIAS,
  WAIT_ASYNC,
  TEMPLATE_FIX,
  REFLECT_INIT,
  REFLECT_DO,
  GPU_REPLICATE,
  GPU_REPLICATE_SYNC,
  GPU_REFLECT,
  GPU_BARRIER,
  GPU_LOOP,

  // clause
  GPU_PRIVATE,
  GPU_FIRSTPRIVATE,
  GPU_MAP_THREADS,
  ;

  private String name = null;
    
  public String getName() {
    if (name == null) name = toString().toLowerCase();

    return name;
  }

  public static XMPpragma valueOf(Xobject x) {
    return valueOf(x.getString());
  }
}
