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
 * Fortran Coarray Object
 */
public class XMPcoarray {

  private class CoarrayInfo {
    Ident body;
    Xtype orgType;
    CoarrayInfo(Ident ident) {
      body = ident;
      orgType = ident.Type().copy();
    }
  }

  private FuncDefBlock def;
  private ArrayList<CoarrayInfo> coarrayList;

  //------------------------------
  //  CONSTRUCTOR
  //------------------------------
  public XMPcoarray(FuncDefBlock def) {
    this.def = def;
    setCoarrayList(def);
  }

  //------------------------------
  //  TRANSLATION
  //------------------------------
  public void run() {
    for (CoarrayInfo info: coarrayList)
      run(info);
  }

  public void run(CoarrayInfo coarray) {

    coarray.body.Type().resetCodimensions();

    ///////////
    System.out.println(display(coarray));    
    ///////////



  }

  //------------------------------
  //  UTILITIES
  //------------------------------
  public ArrayList<CoarrayInfo> getCoarrayList() {
    return coarrayList;
  }

  public void setCoarrayList(FuncDefBlock def) {
    coarrayList = new ArrayList();
    Xobject idList = def.getDef().getFuncIdList();
    for (Xobject obj: (XobjList)idList) {
      Ident ident = (Ident)obj;
      if (ident.Type().getCorank() > 0)
        coarrayList.add(new CoarrayInfo(ident));
    }
  }

  public String toString() {
    String s = "{";
    String delim = "";
    for (CoarrayInfo info: coarrayList) {
      s += delim + toString(info);
      delim = ",";
    }
    return s + "}";
  }
  public String toString(CoarrayInfo info) {
    return toString(info.body);
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

  private String display() {
    String s = "";
    for (CoarrayInfo info: coarrayList)
      s += display(info) + "\n";
    return s;
  }
  private String display(CoarrayInfo info) {
    return "{body:" + toString(info.body)
      + ", orgType:" + toString(info.orgType)
      + "}";
  }
}
