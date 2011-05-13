/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;
import xcodeml.util.XmOption;

import exc.object.*;
import exc.block.*;

public class XMPgpuData {
  public static int GPU_SYNC_IN = 600;
  public static int GPU_SYNC_OUT = 601;

  private String		_name;
  private Ident			_hostDescId;
  private Ident			_hostId;
  private Ident			_deviceDescId;
  private Ident			_deviceAddrId;
  private XMPalignedArray	_alignedArray;
  private XMPtemplate		_alignTemplate;

  public XMPgpuData(String name,
                    Ident hostDescId, Ident hostId,
                    Ident deviceDescId, Ident deviceAddrId,
                    XMPalignedArray alignedArray, XMPtemplate alignTemplate) {
    _name = name;

    _hostDescId = hostDescId;
    _hostId = hostId;
    _deviceDescId = deviceDescId;
    _deviceAddrId = deviceAddrId;

    _alignedArray = alignedArray;
    _alignTemplate = alignTemplate;
  }

  public String getName() {
    return _name;
  }

  public Ident getHostId() {
    return _hostId;
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

  public XMPtemplate getAlignTemplate() {
    return _alignTemplate;
  }

  public static void translateGpuData(PragmaBlock pb, XMPglobalDecl globalDecl) throws XMPexception {
    BlockList gpuDataBody = pb.getBody();

    if (!XmOption.isXcalableMPGPU()) {
      XMP.warning("use -enable-gpu option to use 'gpu data' directive");
      pb.replace(Bcons.COMPOUND(gpuDataBody));
      return;
    }

    // start translation
    XobjList gpuDataDecl = (XobjList)pb.getClauses();
    XMPsymbolTable localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable(pb);
    XMPgpuDataTable gpuDataTable = new XMPgpuDataTable();

    BlockList gpuDataConstructorBody = Bcons.emptyBody();
    BlockList gpuDataDestructorBody = Bcons.emptyBody();
    BlockList replaceBody = Bcons.emptyBody();

    XobjList varList = (XobjList)gpuDataDecl.getArg(0);
    for (XobjArgs i = varList.getArgs(); i != null; i = i.nextArgs()) {
      String varName = i.getArg().getString();

      // check gpuDataTable in the same pragma
      XMPgpuData gpuData = gpuDataTable.getXMPgpuData(varName);
      if (gpuData != null) {
        throw new XMPexception("gpu data '" + varName + "' is already declared");
      }

      XMPpair<Ident, Xtype> typedSpec = XMPutil.findTypedVar(varName, pb);
      Ident varId = typedSpec.getFirst();
      Xtype varType = typedSpec.getSecond();

      // check outer gpuDataTable
      gpuData = XMPgpuDataTable.findXMPgpuData(varName, pb.getParentBlock());
      if (gpuData != null) {
        if (gpuData.getHostId() == varId) {
          throw new XMPexception("gpu data '" + varName + "' is already declared");
        }
      }

      Ident gpuDataHostDescId = replaceBody.declLocalIdent(XMP.GPU_HOST_DESC_PREFIX_ + varName, Xtype.voidPtrType);
      Ident gpuDataDeviceDescId = replaceBody.declLocalIdent(XMP.GPU_DEVICE_DESC_PREFIX_ + varName, Xtype.voidPtrType);
      Ident gpuDataDeviceAddrId = replaceBody.declLocalIdent(XMP.GPU_DEVICE_ADDR_PREFIX_ + varName, Xtype.voidPtrType);

      XMPtemplate alignTemplate = null;
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
                  throw new XMPexception("array '" + varName + "' has has a wrong data type for gpu data");
              }

              addrObj = varId.Ref();
              sizeObj = Xcons.binaryOp(Xcode.MUL_EXPR, Xcons.LongLongConstant(0, XMPutil.getArrayElmtCount(arrayVarType)),
                                                       Xcons.SizeOf(((ArrayType)varType).getArrayElementType()));
            } break;
          default:
            throw new XMPexception("'" + varName + "' has a wrong data type for gpu data");
        }

        gpuDataConstructorBody.add(globalDecl.createFuncCallBlock("_XMP_gpu_init_data_NOT_ALIGNED",
                                                                  Xcons.List(gpuDataHostDescId.getAddr(), gpuDataDeviceDescId.getAddr(), gpuDataDeviceAddrId.getAddr(),
                                                                             addrObj, sizeObj)));
      } else {
        alignTemplate = alignedArray.getAlignTemplate();
        gpuDataConstructorBody.add(globalDecl.createFuncCallBlock("_XMP_gpu_init_data_ALIGNED",
                                                                  Xcons.List(gpuDataHostDescId.getAddr(), gpuDataDeviceDescId.getAddr(), gpuDataDeviceAddrId.getAddr(),
                                                                             alignedArray.getAddrIdVoidRef(), alignedArray.getDescId().Ref())));
      }

      gpuDataDestructorBody.add(globalDecl.createFuncCallBlock("_XMP_gpu_finalize_data", Xcons.List(gpuDataHostDescId.Ref())));

      gpuDataTable.putXMPgpuData(new XMPgpuData(varName,
                                                gpuDataHostDescId, varId,
                                                gpuDataDeviceDescId, gpuDataDeviceAddrId,
                                                alignedArray, alignTemplate));
    }

    replaceBody.add(Bcons.COMPOUND(gpuDataConstructorBody));
    replaceBody.add(Bcons.COMPOUND(gpuDataBody));
    replaceBody.add(Bcons.COMPOUND(gpuDataDestructorBody));
    Block replaceBlock = Bcons.COMPOUND(replaceBody);
    replaceBlock.setProp(XMPgpuDataTable.PROP, gpuDataTable);

    pb.replace(replaceBlock);
  }

  public static void translateGpuSync(PragmaBlock pb, XMPglobalDecl globalDecl) throws XMPexception {
    if (!XmOption.isXcalableMPGPU()) {
      XMP.warning("use -enable-gpu option to use 'gpu sync' directive");
      return;
    }

    // start translation
    XobjList gpuSyncDecl = (XobjList)pb.getClauses();
    XMPsymbolTable localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable(pb);

    BlockList replaceBody = Bcons.emptyBody();

    Xobject directionArg = null;
    String clause = gpuSyncDecl.getArg(1).getString();
    if (clause.equals("in")) {
      directionArg = Xcons.IntConstant(XMPgpuData.GPU_SYNC_IN);
    } else if (clause.equals("out")) {
      directionArg = Xcons.IntConstant(XMPgpuData.GPU_SYNC_OUT);
    } else {
      throw new XMPexception("unknown clause for 'gpu sync'");
    }

    XobjList varList = (XobjList)gpuSyncDecl.getArg(0);
    for (XobjArgs i = varList.getArgs(); i != null; i = i.nextArgs()) {
      String varName = i.getArg().getString();
      XMPgpuData gpuData = XMPgpuDataTable.findXMPgpuData(varName, pb);
      if (gpuData == null) {
        throw new XMPexception("gpu data '" + varName + "' is not declared");
      } else {
        replaceBody.add(globalDecl.createFuncCallBlock("_XMP_gpu_sync", Xcons.List(gpuData.getHostDescId().Ref(), directionArg)));
      }
    }

    pb.replace(Bcons.COMPOUND(replaceBody));
  }
}
