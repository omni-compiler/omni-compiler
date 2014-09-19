package exc.xcalablemp;

import java.util.HashMap;

public class XACCsymbolTable {
  private HashMap<String, XACCdevice> _XACCdeviceTable;
  private HashMap<String, XACCdeviceArray> _XACCdeviceArrayTable;
  
  public XACCsymbolTable() {
    _XACCdeviceTable = new HashMap<String, XACCdevice>();
    _XACCdeviceArrayTable = new HashMap<String, XACCdeviceArray>();
  }

  public void putXMPdevice(XACCdevice o) {
    _XACCdeviceTable.put(o.getName(), o);
  }

  public XACCdevice getXMPdevice(String name) {
    return _XACCdeviceTable.get(name);
  }

  public void putXACCdeviceArray(XACCdeviceArray array) {
    _XACCdeviceArrayTable.put(array.getName(), array);
  }

  public XACCdeviceArray getXACCdeviceArray(String name) {
    return _XACCdeviceArrayTable.get(name);
  }
}
