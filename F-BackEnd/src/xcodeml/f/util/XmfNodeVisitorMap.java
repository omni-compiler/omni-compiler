/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.f.util;

import java.util.HashMap;

/**
 * XcodeML node name to node processing object (visitor) map.
 */
public class XmfNodeVisitorMap<T> {
    public static class Pair<T> {
        public String nodeName;
        public T visitor;
        public Pair(String nodeName, T visitor) {
            this.nodeName = nodeName;
            this.visitor = visitor;
        }
    }

    public XmfNodeVisitorMap(Pair<T>[] pairs) {
        visitorMap = new HashMap<String, T>();
        for (Pair<T> p : pairs) {
            visitorMap.put(p.nodeName, p.visitor);
        }
    }

    public T getVisitor(String nodeName) {
        return visitorMap.get(nodeName);
    }

    private HashMap<String, T> visitorMap;
}
