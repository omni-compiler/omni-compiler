/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

import exc.object.*;
import java.io.*;

public class XMPgpuDecompileWriter extends PrintWriter {
  private XobjectFile _env = null;

  public XMPgpuDecompileWriter(XobjectFile env) {
    super(new OutputStreamWriter(System.out));
    _env = env;
  }

  public void print(XobjectDef def) {
    System.out.println(def.getDef().toString());
  }
}
