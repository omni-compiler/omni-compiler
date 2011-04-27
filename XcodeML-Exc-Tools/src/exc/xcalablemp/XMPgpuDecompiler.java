/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

import exc.object.*;
import java.io.*;

public class XMPgpuDecompiler {
  public static void decompile(XobjectDef def, XobjectFile env) throws XMPexception {
    // FIXME
    try {
      Writer w = new BufferedWriter(new FileWriter("gpu_tmp.c"), 4096);
      XMPgpuDecompileWriter out = new XMPgpuDecompileWriter(w, env);

      out.print(def);
      out.println();
      out.flush();

      w.close();
    } catch (IOException e) {
      throw new XMPexception("error in gpu decompiler: " + e.getMessage());
    }
  }
}
