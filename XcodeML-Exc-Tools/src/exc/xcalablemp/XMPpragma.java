/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

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
  TASK,
  TASKS,
  LOOP,
  REFLECT,
  GMOVE,
  BARRIER,
  REDUCTION,
  BCAST,
  COARRAY,
  GPUDATA,
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
