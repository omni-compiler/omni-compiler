/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

import exc.object.*;
import java.util.Iterator; 

public class XMPrealloc implements XobjectDefVisitor {
  private XMPglobalDecl _globalDecl;

  public XMPrealloc(XMPglobalDecl globalDecl) {
    _globalDecl = globalDecl;
  }

  public void doDef(XobjectDef def) {
    try {
      realloc(def);
    } catch (XMPexception e) {
      // FIXME get line number
      XMP.error(e.getMessage());
    }
  }

  public void realloc(XobjectDef def) throws XMPexception {
    if (def.isVarDecl()) {
      String varName = def.getName();
      XMPalignedArray alignedArray = _globalDecl.getXMPalignedArray(varName);
      if (alignedArray != null) {
        if (alignedArray.realloc()) {
          XobjList allocFuncArgs = Xcons.List(alignedArray.getAddrId().getAddr(), alignedArray.getDescId().Ref());
          for (int i = alignedArray.getDim() - 1; i >= 0; i--) {
            allocFuncArgs.add(Xcons.Cast(Xtype.unsignedlonglongType,
                                         alignedArray.getAccIdAt(i).getAddr()));
          }

          if (alignedArray.getAddrId().getStorageClass() == StorageClass.EXTERN) {
            _globalDecl.addGlobalInitFuncCall("_XMP_init_array_alloc_params", allocFuncArgs);
          }
          else {
            _globalDecl.addGlobalInitFuncCall("_XMP_alloc_array", allocFuncArgs);
          }

          def.setDef(Xcons.List(Xcode.TEXT,
                                Xcons.String("/* array '" + varName + "' is removed by XcalableMP align directive */")));
        }
        else {
          XobjList allocFuncArgs = Xcons.List(alignedArray.getAddrId().getAddr(),
                                              alignedArray.getArrayId().Ref(),
                                              alignedArray.getDescId().Ref());
          for (int i = alignedArray.getDim() - 1; i >= 0; i--) {
            allocFuncArgs.add(Xcons.Cast(Xtype.unsignedlonglongType,
                                         alignedArray.getAccIdAt(i).getAddr()));
          }

          _globalDecl.addGlobalInitFuncCall("_XMP_init_array_addr", allocFuncArgs);
        }
      }
    }
  }
}
