/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

import exc.object.*;

public class XMPgpudata {
  private String		_name;
  private Ident			_descId;
  private boolean		_isAlignedArray;
  private XMPalignedArray	_alignedArray;

  public XMPgpudata(String name, Ident descId, XMPalignedArray alignedArray) {
    _name = name;
    _descId = descId;
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

  public Ident getDescId() {
    return _descId;
  }

  public boolean isAlignedArray() {
    return _isAlignedArray;
  }

  public XMPalignedArray getAlignedArray() {
    return _alignedArray;
  }
}
