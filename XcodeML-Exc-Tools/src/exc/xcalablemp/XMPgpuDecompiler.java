/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

import exc.block.*;
import exc.object.*;
import java.io.*;
import java.util.*;

public class XMPgpuDecompiler {
  public static final String GPU_FUNC_CONF = "XCALABLEMP_GPU_FUNC_CONF_PROP";

  private static XMPgpuDecompileWriter out = null;
  private static final int BUFFER_SIZE = 4096;
  private static final String GPU_SRC_EXTENSION = ".cu";

  public static void decompile(Ident id, XobjList paramIdList, XobjList localVarIdList, CforBlock loopBlock, XobjectFile env) throws XMPexception {
    BlockList loopBody = loopBlock.getBody();

    // schedule iteration
    Iterator<Xobject> iter = localVarIdList.iterator();
    while(iter.hasNext()) {
      Ident localVarId = (Ident)iter.next();
      loopBody.addIdent(localVarId);
      Xobject loopDecls = loopBody.getDecls();
      if (loopDecls == null) {
        loopDecls = Xcons.List();
        loopBody.setDecls(loopDecls);
      }

      loopDecls.add(Xcons.List(Xcode.VAR_DECL, localVarId, null, null));

      XobjList loopIter = XMPutil.getLoopIter(loopBlock, localVarId.getName());
      if (loopIter != null) {
        loopBody.insert(createFuncCallBlock("_XMP_gpu_calc_thread_id", Xcons.List(localVarId.getAddr())));
      }
    }

    Xobject deviceBodyObj = loopBody.toXobject();

    try {
      if (out == null) {
        Writer w = new BufferedWriter(new FileWriter(getSrcName(env.getSourceFileName()) + GPU_SRC_EXTENSION), BUFFER_SIZE);
        out = new XMPgpuDecompileWriter(w, env);
      }

      // add header include line
      out.println("#include \"xmp_gpu_func.hpp\"");
      out.println();

      // decompile device function
      XobjectDef deviceDef = XobjectDef.Func(id, paramIdList, null, deviceBodyObj);
      out.printDeviceFunc(deviceDef, id);
      out.println();

      // generate wrapping function
      Ident hostFuncId = XMP.getMacroId(id.getName() + "_DEVICE");
      Xobject hostFuncCall = hostFuncId.Call(genFuncArgs(paramIdList));
      // FIXME add configuration parameters
      hostFuncCall.setProp(GPU_FUNC_CONF, (Object)Xcons.List());

      // decompie wrapping function
      Xobject hostBodyObj = Xcons.CompoundStatement(hostFuncCall);
      XobjectDef hostDef = XobjectDef.Func(id, paramIdList, null, hostBodyObj);
      out.printHostFunc(hostDef);
      out.println();

      out.flush();
    } catch (IOException e) {
      throw new XMPexception("error in gpu decompiler: " + e.getMessage());
    }
  }

  private static Block createFuncCallBlock(String funcName, XobjList funcArgs) {
    Ident funcId = XMP.getMacroId(funcName);
    return Bcons.Statement(funcId.Call(funcArgs));
  }

  private static String getSrcName(String srcName) {
    String name = "";
    String[] buffer = srcName.split("\\.");
    for (int i = 0; i < buffer.length - 1; i++) {
      name += buffer[i];
    }

    return name;
  }

  private static XobjList genFuncArgs(XobjList paramIdList) {
    XobjList funcArgs = Xcons.List();
    for (XobjArgs i = paramIdList.getArgs(); i != null; i = i.nextArgs()) {
      Ident id = (Ident)i.getArg();
      funcArgs.add(id.Ref());
    }

    return funcArgs;
  }
}
