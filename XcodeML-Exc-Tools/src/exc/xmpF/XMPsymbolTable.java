/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xmpF;

import java.util.HashMap;
import java.util.Vector;

public class XMPsymbolTable {
  private HashMap<String, XMPobject> objectTable;
  private HashMap<String, XMParray> arrayTable;
  private HashMap<String, XMPcoarray> coarrayTable;

  private Vector<XMPobject> objects;
  private Vector<XMParray> arrays;
  private Vector<XMPcoarray> coarrays;

  public XMPsymbolTable() {
    objectTable = new HashMap<String, XMPobject>();
    arrayTable = new HashMap<String, XMParray>();
    coarrayTable = new HashMap<String, XMPcoarray>();

    objects = new Vector<XMPobject>();
    arrays = new Vector<XMParray>();
    coarrays = new Vector<XMPcoarray>();
  }

  /*
   * objects (nodes and templates)
   */
  public void putXMPobject(XMPobject o) {
    objectTable.put(o.getName(), o);
    objects.add(o);
  }

  public XMPobject getXMPobject(String name) {
    return objectTable.get(name);
  }

  public XMPnodes getXMPnodes(String name) {
    XMPobject o = objectTable.get(name);

    if (o == null) return null;
    else if (o.getKind() != XMPobject.NODES) return null;
    else return (XMPnodes)o;
  }

  public XMPtemplate getXMPtemplate(String name) {
    XMPobject o = objectTable.get(name);

    if (o == null) return null;
    else if (o.getKind() != XMPobject.TEMPLATE) return null;
    else return (XMPtemplate)o;
  }

  public Vector<XMPobject> getXMPobjects(){
    return objects;
  }

  /*
   * array
   */
  public void putXMParray(XMParray array) {
    arrayTable.put(array.getName(), array);
    arrays.add(array);
  }

  public XMParray getXMParray(String name) {
    return arrayTable.get(name);
  }

  public Vector<XMParray> getXMParrays(){
    return arrays;
  }

  /*
   * coarray
   */
  public void putXMPcoarray(XMPcoarray array) {
    coarrayTable.put(array.getName(), array);
    coarrays.add(array);
  }

  public XMPcoarray getXMPcoarray(String name) {
    return coarrayTable.get(name);
  }

  public Vector<XMPcoarray> getXMPcoarrys(){
    return coarrays;
  }
}
