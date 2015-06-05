/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xmpF;

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
    LOCAL_ALIAS,
    SAVE_DESC,
    TEMPLATE_FIX,
    TASK,
    TASKS,
    LOOP,
    REFLECT,
    GMOVE,
    BARRIER,
    REDUCTION,
    BCAST,
    WAIT_ASYNC,
    COARRAY,
    ARRAY,

    FUNCTION_BODY,      
    PRAGMA_END;

  private String name = null;
    
  public String getName() {
    if (name == null) name = toString().toLowerCase();

    return name;
  }

  public static XMPpragma valueOf(Xobject x) {
    return valueOf(x.getString());
  }
}
