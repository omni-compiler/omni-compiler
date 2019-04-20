package exc.xmpF;

import java.util.HashMap;
import java.util.Vector;

/**
 * Object to represent the symbol table in XMP analysis.
 *  It contains XMP objects, XMP global arrays, coarrays.
 *  These objects can be retrived by its name.
 */

public class XMPsymbolTable {
  private HashMap<String, XMPobject> objectTable;
  private HashMap<String, XMParray> arrayTable;
  private HashMap<String, XMPcoarray> coarrayTable;

  private Vector<XMPobject> objects;
  private Vector<XMParray> arrays;
  private Vector<XMPcoarray> coarrays;
  
  private Vector<String> used_modules;

  /** 
   * Contructor: construct an empty symbol tables.
   */
  public XMPsymbolTable() {
    objectTable = new HashMap<String, XMPobject>();
    arrayTable = new HashMap<String, XMParray>();
    coarrayTable = new HashMap<String, XMPcoarray>();

    objects = new Vector<XMPobject>();
    arrays = new Vector<XMParray>();
    coarrays = new Vector<XMPcoarray>();

    used_modules = new Vector<String>();
  }

  /**
   * Add the given module name in the used module name list.
   *  This maintains use-used relationship between modules.
   */
  public void addUseModule(String module_name){
    used_modules.add(module_name);
  }

  /**
   * return the list of used modules as the vector of the used module name.
   */
  public Vector<String> getUsedModules(){
    return used_modules;
  }

  /*
   * objects (nodes and templates)
   */
  /**
   * put the give XMPobjct to the symbol table.
   * The XMPobject can be retrived by the name using getXMPobject method.
   */
  public void putXMPobject(XMPobject o) {
    objectTable.put(o.getName(), o);
    objects.add(o);
  }

  /**
   * Retrive the XMPobject by the give name.
   */
  public XMPobject getXMPobject(String name) {
    return objectTable.get(name);
  }

  /**
   * return all XMPobject stored in this symbol table.
   */
  public Vector<XMPobject> getXMPobjects(){
    return objects;
  }

  /*
   * array
   */
  /**
   * put the give XMParray to the symbol table.
   * The XMParray can be retrived by the name using getXMParray method.  
   */
  public void putXMParray(XMParray array) {
    arrayTable.put(array.getName(), array);
    arrays.add(array);
  }

  /**
   * Retrive the XMParray by the give name.
   */
  public XMParray getXMParray(String name) {
    return arrayTable.get(name);
  }

  /**
   * return all XMParray stored in this symbol table.
   */
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
