package exc.xcalablemp;

import java.util.HashMap;

public class XMPobjectTable {
  private HashMap<String, XMPobject> _objectTable;
  private HashMap<String, XMPalignedArray> _alignedArrayTable;
  //private HashMap<String, XMPcoarray> coarrayTable;
  //private int coarrayCount;

  public XMPobjectTable() {
    _objectTable = new HashMap<String, XMPobject>();
    _alignedArrayTable = new HashMap<String, XMPalignedArray>();
    //coarrayTable = new HashMap<String, XMPcoarray>();
    //coarrayCount = 0;
  }

  public void putObject(XMPobject o) {
    _objectTable.put(o.getName(), o);
  }

  public XMPobject getObject(String name) {
    return _objectTable.get(name);
  }

  public XMPnodes getNodes(String name) {
    XMPobject o = _objectTable.get(name);

    if (o == null) return null;
    else if (o.getKind() != XMPobject.NODES) return null;
    else return (XMPnodes)o;
  }

  public XMPtemplate getTemplate(String name) {
    XMPobject o = _objectTable.get(name);

    if (o == null) return null;
    else if (o.getKind() != XMPobject.TEMPLATE) return null;
    else return (XMPtemplate)o;
  }

  public void putAlignedArray(XMPalignedArray alignedArray) {
    _alignedArrayTable.put(alignedArray.getName(), alignedArray);
  }

  public XMPalignedArray getAlignedArray(String name) {
    return _alignedArrayTable.get(name);
  }

  /*
  public void putCoarray(XMPcoarray c)
  {
    coarrayTable.put(c.getName(), c);
  }

  public XMPcoarray getCoarray(String name)
  {
    return coarrayTable.get(name);
  }

  public int increaseCoarrayCount()
  {
    return coarrayCount++;
  }

  public int getCoarrayCount()
  {
    return coarrayCount;
  }
  */
}
