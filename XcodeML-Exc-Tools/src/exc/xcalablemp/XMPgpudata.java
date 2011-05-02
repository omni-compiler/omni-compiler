/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

import exc.object.*;

public class XMPgpudata {
  public static int GPUSYNC_IN = 600;
  public static int GPUSYNC_OUT = 601;

  private String		_name;
  private Ident			_hostDescId;
  private Ident			_deviceDescId;
  private Ident			_deviceAddrId;
  private boolean		_isAlignedArray;
  private XMPalignedArray	_alignedArray;

  public XMPgpudata(String name, Ident hostDescId, Ident deviceDescId, Ident deviceAddrId,
                    XMPalignedArray alignedArray) {
    _name = name;

    _hostDescId = hostDescId;
    _deviceDescId = deviceDescId;
    _deviceAddrId = deviceAddrId;

    if (alignedArray == null) {
      _isAlignedArray = false;
    } else {
      _isAlignedArray = true;
    }
    _alignedArray = alignedArray;
  }

  public String getName() {
    return _name;
  }

  public Ident getHostDescId() {
    return _hostDescId;
  }

  public Ident getDeviceDescId() {
    return _deviceDescId;
  }

  public Ident getDeviceAddrId() {
    return _deviceAddrId;
  }

  public boolean isAlignedArray() {
    return _isAlignedArray;
  }

  public XMPalignedArray getAlignedArray() {
    return _alignedArray;
  }
}
