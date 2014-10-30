/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xmpF;

import exc.object.*;
import exc.block.*;
import java.util.*;

/*
 * Coarray Object
 */
public class XMPcoarray {

  private Ident ident;
  Xtype orgType;

  //------------------------------
  //  CONSTRUCTOR
  //------------------------------
  public XMPcoarray(Ident ident) {
    this.ident = ident;
    orgType = ident.Type().copy();
  }

  //------------------------------
  //  UTILITIES
  //------------------------------
  public Boolean isScalar() {
    return (ident.Type().getNumDimensions() == 0);
  }

  public Boolean isAllocatable() {
    return ident.Type().isFallocatable();
  }

  public Boolean isPointer() {
    return ident.Type().isFpointer();
  }

  public Boolean isAssumedSize() {
    return ident.Type().isFassumedSize();
  }

  public Boolean isAssumedShape() {
    return ident.Type().isFassumedShape();
  }

  public Boolean isExplicitShape() {
    return (!isAssumedSize() && !isAssumedShape() &&
            !isAllocatable() && !isPointer());
  }


  public Ident getIdent() {
    return ident;
  }

  public void setIdent(Ident ident) {
    this.ident = ident;
  }

  public Xobject[] getCodimensions() {
    return ident.Type().getCodimensions();
  }

  public void setCodimensions(Xobject[] codimensions) {
    ident.Type().setCodimensions(codimensions);
  }

  public void resetCodimensions() {
    ident.Type().resetCodimensions();
  }

  public String getName() {
    return ident.getName();
  }

  public String toString() {
    return toString(ident);
  }
  public String toString(Xobject obj) {
    return "Xobject(" + obj.getName()
      + ",rank=" + obj.Type().getNumDimensions()
      + ",corank=" + obj.Type().getCorank()
      + ")";
  }
  public String toString(Xtype type) {
    return "Xtype(rank=" + type.getNumDimensions()
      + ",corank=" + type.getCorank()
      + ")";
  }

  public String display() {
    return "{ident:" + toString(ident)
      + ", orgType:" + toString(orgType)
      + "}";
  }
}

