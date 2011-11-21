/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.f.decompile;

import java.util.Deque;
import java.util.LinkedList;

import org.w3c.dom.Node;

/**
 * Decompile utility, using DOM node.
 */
public class XfUtilForDom {
    public static void errorPrint(String message) {
        System.err.println(message);
    }

    public static String[] getNodePath(Node node) {
        Deque<String> list = new LinkedList<String>();
        list.addFirst(node.getNodeName());
        node = node.getParentNode();
        while (node != null) {
            list.addFirst(node.getNodeName());
            node = node.getParentNode();
        }
        return list.toArray(new String[0]);
    }

    public static String formatError(Node errorNode,
                                     XfError errorCode,
                                     Object... args) {
        StringBuilder sb = new StringBuilder();
        sb.append("Error(");
        sb.append(errorCode.ordinal());
        sb.append("): ");
        sb.append(errorCode.format(args));

        if (errorNode == null) {
            return sb.toString();
        }

        sb.append("\n<Element Tree>\n");

        String[] nodeArray = getNodePath(errorNode);
        if (nodeArray.length > 0) {
            sb.append(nodeArray[0]);
            sb.append("\n");
        }

        for (int i = 1; i < nodeArray.length; i++) {
            for (int j = 0; j < i; j++) {
                sb.append("  ");
            }
            sb.append("+");
            sb.append(nodeArray[i]);
            sb.append("\n");
        }
        return sb.toString();
    }

    public static boolean isNullOrEmpty(String s) {
        return ((s == null) || (s.isEmpty() != false));
    }
}
