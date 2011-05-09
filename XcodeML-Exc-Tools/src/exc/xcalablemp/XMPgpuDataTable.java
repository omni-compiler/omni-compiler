/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

import exc.block.*;
import java.util.HashMap;

public class XMPgpuDataTable {
  public static String PROP = "XMP_GPU_DATA_TABLE";

  private HashMap<String, XMPgpuData> _XMPgpuDataTable;

  public XMPgpuDataTable() {
    _XMPgpuDataTable = new HashMap<String, XMPgpuData>();
  }

  public void putXMPgpuData(XMPgpuData gpuData) {
    _XMPgpuDataTable.put(gpuData.getName(), gpuData);
  }

  public XMPgpuData getXMPgpuData(String name) {
    return _XMPgpuDataTable.get(name);
  }

  public static XMPgpuData findXMPgpuData(String varName, Block block) {
    if (block == null) {
      return null;
    }

    for (Block b = block; b != null; b = b.getParentBlock()) {
      XMPgpuDataTable gpuDataTable = (XMPgpuDataTable)b.getProp(XMPgpuDataTable.PROP);
      if (gpuDataTable != null) {
        XMPgpuData gpuData = gpuDataTable.getXMPgpuData(varName);
        if (gpuData != null) {
          return gpuData;
        }
      }
    }

    return null;
  }
}
