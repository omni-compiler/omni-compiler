/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.util;

import org.w3c.dom.*;
import java.util.ArrayList;

public class XmDomUtil {
    public static Node getElement(Node n, String name) {
        if (n == null)
            return null;

        NodeList list = n.getChildNodes();
        for (int i = 0; i < list.getLength(); i++) {
            Node thisNode = list.item(i);
            if (thisNode.getNodeType() != Node.ELEMENT_NODE)
                continue;
            if (thisNode.getNodeName() == name)
                return thisNode;
        }
        return null;
    }

    public static Node getContent(Node n) {
        if (n == null)
            return null;

        NodeList list = n.getChildNodes();
        for (int i = 0; i < list.getLength(); i++) {
            Node thisNode = list.item(i);
            if (thisNode.getNodeType() != Node.ELEMENT_NODE)
                continue;
            return thisNode;
        }
        return null;
    }

    public static String getContentText(Node n) {
        if (n == null)
            return null;
        if (n.getFirstChild() == null) {
            return "";
        }

        return n.getFirstChild().getNodeValue();
    }

    public static String getAttr(Node n, String name) {
        if (n == null)
            return null;

        Node nn = n.getAttributes().getNamedItem(name);
        if (nn != null)
            return nn.getNodeValue();
        return null;
    }

    public static boolean getAttrBool(Node n, String name) {
        if (n == null)
            return false;

        Node nn = n.getAttributes().getNamedItem(name);
        if (nn != null)
            return true;
        return false;
    }

    public static Node[] collectElements(Node n, String name) {
        if (n == null) {
            return new Node[0];
        }

        NodeList list = n.getChildNodes();
        ArrayList<Node> nodes = new ArrayList<Node>();
        for (int i = 0; i < list.getLength(); i++) {
            Node childNode = list.item(i);
            if (childNode.getNodeType() != Node.ELEMENT_NODE)
                continue;
            if (childNode.getNodeName().equals(name))
                nodes.add(childNode);
        }
        return nodes.toArray(new Node[0]);
    }

    public static ArrayList<Node> collectChildNodes(Node n) {
        if (n == null) {
            return null;
        }

        NodeList list = n.getChildNodes();
        ArrayList<Node> nodes = new ArrayList<Node>();
        for (int i = 0; i < list.getLength(); i++) {
            Node childNode = list.item(i);
            if (childNode.getNodeType() != Node.ELEMENT_NODE)
                continue;
            nodes.add(childNode);
        }
        return nodes;
    }
}
