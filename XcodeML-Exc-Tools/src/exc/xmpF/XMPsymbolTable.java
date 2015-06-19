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
  
  private Vector<String> used_modules;

  public XMPsymbolTable() {
    objectTable = new HashMap<String, XMPobject>();
    arrayTable = new HashMap<String, XMParray>();
    coarrayTable = new HashMap<String, XMPcoarray>();

    objects = new Vector<XMPobject>();
    arrays = new Vector<XMParray>();
    coarrays = new Vector<XMPcoarray>();

    used_modules = new Vector<String>();
  }

  /* module */
  public void addUseModule(String module_name){
    used_modules.add(module_name);
  }

  public Vector<String> getUsedModules(){
    return used_modules;
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


  // for debug
  public void dump(String msg){
    System.out.println("dump XMP Symbol Table: --- "+msg);
    for(XMPobject o: objects) System.out.println("o="+o);
    for(XMParray a: arrays) System.out.println("a="+a);
    System.out.println("dump end: ---");
  }
}
