/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

import java.util.HashMap;

public class XMPgpudataTable {
  public static String PROP = "XMP_GPU_DATA_TABLE";

  private HashMap<String, XMPgpudata> _XMPgpudataTable;

  public XMPgpudataTable() {
    _XMPgpudataTable = new HashMap<String, XMPgpudata>();
  }

  public void putXMPgpudata(XMPgpudata gpudata) {
    _XMPgpudataTable.put(gpudata.getName(), gpudata);
  }

  public XMPgpudata getXMPgpudata(String name) {
    return _XMPgpudataTable.get(name);
  }
}
