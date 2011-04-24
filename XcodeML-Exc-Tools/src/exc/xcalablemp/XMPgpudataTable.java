/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

import exc.block.*;
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

  public static XMPgpudata findXMPgpudata(String varName, Block block) {
    if (block == null) {
      return null;
    }

    for (Block b = block; b != null; b = b.getParentBlock()) {
      XMPgpudataTable gpudataTable = (XMPgpudataTable)b.getProp(XMPgpudataTable.PROP);
      if (gpudataTable != null) {
        XMPgpudata gpudata = gpudataTable.getXMPgpudata(varName);
        if (gpudata != null) {
          return gpudata;
        }
      }
    }

    return null;
  }
}
