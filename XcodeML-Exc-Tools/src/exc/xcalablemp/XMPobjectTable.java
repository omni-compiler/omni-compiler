/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

import java.util.HashMap;

public class XMPobjectTable {
  private HashMap<String, XMPobject> _objectTable;
  private HashMap<String, XMPalignedArray> _alignedArrayTable;

  public XMPobjectTable() {
    _objectTable = new HashMap<String, XMPobject>();
    _alignedArrayTable = new HashMap<String, XMPalignedArray>();
  }

  public void putXMPobject(XMPobject o) {
    _objectTable.put(o.getName(), o);
  }

  public XMPobject getXMPobject(String name) {
    return _objectTable.get(name);
  }

  public XMPnodes getXMPnodes(String name) {
    XMPobject o = _objectTable.get(name);

    if (o == null) return null;
    else if (o.getKind() != XMPobject.NODES) return null;
    else return (XMPnodes)o;
  }

  public XMPtemplate getXMPtemplate(String name) {
    XMPobject o = _objectTable.get(name);

    if (o == null) return null;
    else if (o.getKind() != XMPobject.TEMPLATE) return null;
    else return (XMPtemplate)o;
  }

  public void putXMPalignedArray(XMPalignedArray alignedArray) {
    _alignedArrayTable.put(alignedArray.getName(), alignedArray);
  }

  public XMPalignedArray getXMPalignedArray(String name) {
    return _alignedArrayTable.get(name);
  }
}
