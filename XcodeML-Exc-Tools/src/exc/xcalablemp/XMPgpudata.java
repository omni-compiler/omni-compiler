/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;
import xcodeml.util.XmOption;

import exc.object.*;
import exc.block.*;

public class XMPgpudata {
  public static int GPUSYNC_IN = 600;
  public static int GPUSYNC_OUT = 601;

  private String		_name;
  private Ident			_hostDescId;
  private Ident			_deviceDescId;
  private Ident			_deviceAddrId;
  private XMPalignedArray	_alignedArray;

  public XMPgpudata(String name, Ident hostDescId, Ident deviceDescId, Ident deviceAddrId,
                    XMPalignedArray alignedArray) {
    _name = name;

    _hostDescId = hostDescId;
    _deviceDescId = deviceDescId;
    _deviceAddrId = deviceAddrId;

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

  public XMPalignedArray getXMPalignedArray() {
    return _alignedArray;
  }

  public static void translateGpudata(PragmaBlock pb, XMPglobalDecl globalDecl) throws XMPexception {
    BlockList gpudataBody = pb.getBody();

    if (!XmOption.isXcalableMPGPU()) {
      XMP.warning("use -enable-gpu option to use 'gpudata' directive");
      pb.replace(Bcons.COMPOUND(gpudataBody));
      return;
    }

    // start translation
    XobjList gpudataDecl = (XobjList)pb.getClauses();
    XMPsymbolTable localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable(pb);
    XMPgpudataTable gpudataTable = new XMPgpudataTable();

    BlockList gpudataConstructorBody = Bcons.emptyBody();
    BlockList gpudataDestructorBody = Bcons.emptyBody();

    BlockList replaceBody = Bcons.blockList(Bcons.COMPOUND(gpudataConstructorBody),
                                            Bcons.COMPOUND(gpudataBody),
                                            Bcons.COMPOUND(gpudataDestructorBody));

    XobjList varList = (XobjList)gpudataDecl.getArg(0);
    for (XobjArgs i = varList.getArgs(); i != null; i = i.nextArgs()) {
      String varName = i.getArg().getString();

      // FIXME check gpudataTable FIXME do not allow gpudata to be nested
      XMPgpudata gpudata = gpudataTable.getXMPgpudata(varName);
      if (gpudata != null) {
        throw new XMPexception("gpudata '" + varName + "' is already declared");
      }

      XMPpair<Ident, Xtype> typedSpec = XMPutil.findTypedVar(varName, pb);
      Ident varId = typedSpec.getFirst();
      Xtype varType = typedSpec.getSecond();

      Ident gpudataHostDescId = replaceBody.declLocalIdent(XMP.GPU_HOST_DESC_PREFIX_ + varName, Xtype.voidPtrType);
      Ident gpudataDeviceDescId = replaceBody.declLocalIdent(XMP.GPU_DEVICE_DESC_PREFIX_ + varName, Xtype.voidPtrType);
      Ident gpudataDeviceAddrId = replaceBody.declLocalIdent(XMP.GPU_DEVICE_ADDR_PREFIX_ + varName, Xtype.voidPtrType);

      XMPalignedArray alignedArray = globalDecl.getXMPalignedArray(varName, localXMPsymbolTable);
      if (alignedArray == null) {
        Xobject addrObj = null;
        Xobject sizeObj = null;
        switch (varType.getKind()) {
          case Xtype.BASIC:
          case Xtype.STRUCT:
          case Xtype.UNION:
            {
              addrObj = varId.getAddr();
              sizeObj = Xcons.SizeOf(varType);
            } break;
          case Xtype.ARRAY:
            {
              ArrayType arrayVarType = (ArrayType)varType;
              switch (arrayVarType.getArrayElementType().getKind()) {
                case Xtype.BASIC:
                case Xtype.STRUCT:
                case Xtype.UNION:
                  break;
                default:
                  throw new XMPexception("array '" + varName + "' has has a wrong data type for gpudata");
              }

              addrObj = varId.Ref();
              sizeObj = Xcons.binaryOp(Xcode.MUL_EXPR, Xcons.LongLongConstant(0, XMPutil.getArrayElmtCount(arrayVarType)),
                                                       Xcons.SizeOf(((ArrayType)varType).getArrayElementType()));
            } break;
          default:
            throw new XMPexception("'" + varName + "' has a wrong data type for broadcast");
        }

        gpudataConstructorBody.add(globalDecl.createFuncCallBlock("_XMP_gpu_init_gpudata_NOT_ALIGNED",
                                                                  Xcons.List(gpudataHostDescId.getAddr(), gpudataDeviceDescId.getAddr(), gpudataDeviceAddrId.getAddr(),
                                                                             addrObj, sizeObj)));
      } else {
        gpudataConstructorBody.add(globalDecl.createFuncCallBlock("_XMP_gpu_init_gpudata_ALIGNED",
                                                                  Xcons.List(gpudataHostDescId.getAddr(), gpudataDeviceDescId.getAddr(), gpudataDeviceAddrId.getAddr(),
                                                                             alignedArray.getAddrIdVoidRef(), alignedArray.getDescId().Ref())));
      }

      gpudataDestructorBody.add(globalDecl.createFuncCallBlock("_XMP_gpu_finalize_gpudata", Xcons.List(gpudataHostDescId.Ref())));

      gpudataTable.putXMPgpudata(new XMPgpudata(varName, gpudataHostDescId, gpudataDeviceDescId, gpudataDeviceAddrId, alignedArray));
    }

    Block replaceBlock = Bcons.COMPOUND(replaceBody);
    replaceBlock.setProp(XMPgpudataTable.PROP, gpudataTable);

    pb.replace(replaceBlock);
  }

  public static void translateGpusync(PragmaBlock pb, XMPglobalDecl globalDecl) throws XMPexception {
    if (!XmOption.isXcalableMPGPU()) {
      XMP.warning("use -enable-gpu option to use 'gpusync' directive");
      return;
    }

    // start translation
    XobjList gpusyncDecl = (XobjList)pb.getClauses();
    XMPsymbolTable localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable(pb);

    BlockList replaceBody = Bcons.emptyBody();

    Xobject directionArg = null;
    String clause = gpusyncDecl.getArg(1).getString();
    if (clause.equals("in")) {
      directionArg = Xcons.IntConstant(XMPgpudata.GPUSYNC_IN);
    } else if (clause.equals("out")) {
      directionArg = Xcons.IntConstant(XMPgpudata.GPUSYNC_OUT);
    } else {
      throw new XMPexception("unknown clause for 'gpusync'");
    }

    XobjList varList = (XobjList)gpusyncDecl.getArg(0);
    for (XobjArgs i = varList.getArgs(); i != null; i = i.nextArgs()) {
      String varName = i.getArg().getString();
      XMPgpudata gpudata = XMPgpudataTable.findXMPgpudata(varName, pb);
      if (gpudata == null) {
        throw new XMPexception("gpudata '" + varName + "' is not declared");
      } else {
        replaceBody.add(globalDecl.createFuncCallBlock("_XMP_gpu_sync", Xcons.List(gpudata.getHostDescId().Ref(), directionArg)));
      }
    }

    pb.replace(Bcons.COMPOUND(replaceBody));
  }
}
