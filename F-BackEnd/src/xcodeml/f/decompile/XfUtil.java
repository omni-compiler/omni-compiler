/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.f.decompile;

import java.util.Deque;
import java.util.LinkedList;

import xcodeml.binding.IRNode;

/**
 * Decompile utility.
 */
class XfUtil
{
    public static void errorPrint(String message)
    {
        System.err.println(message);
    }

    /**
     * Convert Relaxer generated class name into an element name.
     * @param node Instance of IRNode
     * @return Element name.
     */
    public static String getElementName(IRNode node)
    {
        String nodeName = node.getClass().getSimpleName();
        if (nodeName.startsWith("Xbf") == false) {
            return nodeName;
        }
        return nodeName.substring(3);
    }

    public static String[] getNodePath(IRNode node)
    {
        Deque<String> list = new LinkedList<String>();
        list.addFirst(getElementName(node));
        node = node.rGetParentRNode();
        while (node != null) {
            list.addFirst(getElementName(node));
            node = node.rGetParentRNode();
        }
        return list.toArray(new String[0]);
    }

    public static String formatError(IRNode errorNode, XfError errorCode, Object... args)
    {
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

    public static boolean isNullOrEmpty(String s)
    {
        return ((s == null) || (s.isEmpty() != false));
    }
}
